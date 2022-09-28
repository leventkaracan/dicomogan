import pytorch_lightning as pl
from experiments_utils import *
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dicomogan.losses.loss_lib import GANLoss, VGGLoss
from dicomogan.losses.clip_loss import CLIPLoss
import os
from PIL import Image
import random
import clip
from PIL import Image
import math

class New_DiCoMOGAN(pl.LightningModule):
    def __init__(self,
                generator_config,
                mapping_config,
                lambda_G,
                scheduler_config = None,
                custom_loggers = None,
                n_critic = 1):
        super(New_DiCoMOGAN, self).__init__()

        self.clip_img_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224), 
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.lambda_G = lambda_G

        # Intializing networks
        self.G = instantiate_from_config(generator_config)
        self.mapping_network = instantiate_from_config(mapping_config)
        
        self.G.eval()
        self.requires_grad(self.G, False)
        
        # Losses
        self.reconstruction_loss = nn.MSELoss()
        self.directional_clip_loss = CLIPLoss()
        
        self.scheduler_config = scheduler_config
        
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
        
    def get_text_embedding(self, description):
        return self.directional_clip_loss.encode_text(self.directional_clip_loss.tokenize(description))
    
    def preprocess_feat(self, latent_feat):
        latent_feat = latent_feat.clone()
        bs = int(latent_feat.size(0)/2)
        latent_feat_mismatch = torch.roll(latent_feat, 1, 0)
        latent_splits = torch.split(latent_feat, bs, 0)
        latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], -1, 0), latent_splits[1]), 0)
        return latent_feat_mismatch, latent_feat_relevant
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        vid = batch['img']
        input_desc = batch['raw_desc']
        sampleT = batch['sampleT']
        inversions = batch['inversion']
        
        if sampleT.shape[0] > 0:
            assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        bs, T, ch, height, width = vid.size()
        
        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        #TODO: Check if needed (Since the actual frames are never passed to an encoder, is this necessary?)
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # T*B x C x H x W // range [-1,1]
        
        txt_feat = self.get_text_embedding(input_desc) # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames, 1, 1) # T x B x D
        txt_feat = txt_feat.view(bs * n_frames, -1) # T * B x D
        
        txt_feat_mismatch, _ = self.preprocess_feat(txt_feat)
        B, n_frames, n_channels, dim = inversions.shape
        
        inversions = inversions.reshape(B * n_frames, n_channels, dim)
        adjusted_latent = inversions + self.mapping_network(inversions, txt_feat)
        adjusted_mismatched_latent = inversions + self.mapping_network(inversions, txt_feat_mismatch)

        mismatched_img = self.G(adjusted_mismatched_latent) #.reshape(bs * T, ch, height, width)
        reconstructed = self.G(adjusted_latent) #.reshape(bs * T, ch, height, width)

        reconstructed = nn.functional.interpolate(reconstructed, size=(256, 192), mode="bicubic", align_corners=False)
        mismatched_img = nn.functional.interpolate(mismatched_img, size=(256, 192), mode="bicubic", align_corners=False)
        
        reconstruction_loss = self.reconstruction_loss(reconstructed, video_sample_norm)
        directional_clip_loss = self.directional_clip_loss(video_sample_norm, txt_feat, mismatched_img, txt_feat_mismatch, self.global_step)
        
        self.log("train/rl_loss", reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/clip_loss", directional_clip_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        total_loss = reconstruction_loss + directional_clip_loss
        
        return total_loss
    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.mapping_network.parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
        
        ret = {"optimizer": optimizer, "frequency": 1}
        
        ae_scheduler = None
        if self.scheduler_config is not None:
            ae_scheduler = LambdaLR(opt_ae, lr_lambda=instantiate_from_config(self.scheduler_config).schedule, verbose=False)
            ret['lr_scheduler'] = {
                        'scheduler': ae_scheduler, 
                        'interval': 'step'}
        
        return ret
    
    def log_images(self, batch, split):
        ret = {}

        vid = batch['img']
        input_desc = batch['raw_desc']
        sampleT = batch['sampleT']
        inversions = batch['inversion']
        
        if sampleT.shape[0] > 0:
            assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        bs, T, ch, height, width = vid.size()
        
        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # T*B x C x H x W // range [-1,1]
        ret['real_images'] = video_sample_norm
        
        txt_feat = self.get_text_embedding(input_desc) # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames, 1, 1) # T x B x D
        txt_feat = txt_feat.view(bs * n_frames, -1) # T * B x D
        
        txt_feat_mismatch, _ = self.preprocess_feat(txt_feat)
        B, n_frames, n_channels, dim = inversions.shape
        
        inversions = inversions.reshape(B * n_frames, n_channels, dim)
        adjusted_latent = inversions + self.mapping_network(inversions, txt_feat)
        adjusted_mismatched_latent = inversions + self.mapping_network(inversions, txt_feat_mismatch)

        mismatched_img = self.G(adjusted_mismatched_latent) #.reshape(bs * T, ch, height, width)
        reconstructed = self.G(adjusted_latent) #.reshape(bs * T, ch, height, width)

        ret['reconstruction'] = reconstructed
        ret['mismatched_text'] = mismatched_img
        return ret
             
        
           
        
        
     
        
                
        
    
        
        
        

        


