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
from dicomogan.modules import ProjNetwork#, VecClassifier
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
                n_attrs,
                content_dim,
                style_dim,
                scheduler_config = None,
                custom_loggers = None,
                delta_inversion_weight = 0.05,
                rec_loss_lambda=1.0,
                l2_latent_lambda=1.0,
                clip_loss_lambda=1.0,
                classifier_loss_lambda=1.0,
                l2_latent_eps = 1.0,
                tgt_text = None,
                n_critic = 1):
        super(New_DiCoMOGAN, self).__init__()

        self.clip_img_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224), 
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        # Intializing networks
        self.G = instantiate_from_config(generator_config)
        self.mapping_network = instantiate_from_config(mapping_config)
        
        self.requires_grad(self.G, False)

        self.content_proj = ProjNetwork(512, content_dim, n_hidden_layers=3)
        self.style_proj = ProjNetwork(512, style_dim, n_hidden_layers=3)
        self.final_proj = ProjNetwork(content_dim, 512, n_hidden_layers=3)

        self.clip_proj = ProjNetwork(512, content_dim, n_hidden_layers=3)

        # TODO: Train and load weights for the classifier
        #self.classifier = VecClassifier(style_dim, n_attrs)

        self.content_dim = content_dim
        self.style_dim = style_dim
        
        # lambdas
        self.rec_loss_lambda = rec_loss_lambda
        self.l2_latent_lambda = l2_latent_lambda
        self.clip_loss_lambda = clip_loss_lambda
        self.classifier_loss_lambda = classifier_loss_lambda
        self.delta_inversion_weight = delta_inversion_weight
        self.l2_latent_eps = l2_latent_eps

        # Losses
        self.reconstruction_loss = nn.MSELoss()
        self.l2_latent_loss = nn.MSELoss()
        self.directional_clip_loss = CLIPLoss()
        self.criterionVGG = VGGLoss()
        
        self.tgt_text_embed = None
        if tgt_text is not None:
            self.tgt_text_embed = self.get_text_embedding([tgt_text]) # 1 x 512
            self.tgt_text_embed.requires_grad = False

        self.scheduler_config = scheduler_config
        
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
        
    def get_text_embedding(self, description):
        return self.directional_clip_loss.encode_text(self.directional_clip_loss.tokenize(description))
    
    def preprocess_text_feat(self, latent_feat, mx_roll=1):
        bs = int(latent_feat.size(0)/2)
        if self.tgt_text_embed is not None:
            self.tgt_text_embed = self.tgt_text_embed.to(latent_feat.device)
            latent_feat_mismatch = self.tgt_text_embed.repeat(latent_feat.size(0), 1)
            latent_splits = torch.split(latent_feat, bs, 0)
            latent_feat_relevant = torch.cat((self.tgt_text_embed.repeat(bs, 1), latent_splits[1]), 0)
        else:
            if mx_roll > 1:
                roll_seed = np.random.randint(1, mx_roll)
            else:
                roll_seed = 1
            latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=0)
            latent_splits = torch.split(latent_feat, bs, 0)

            if mx_roll > 1:
                roll_seed = np.random.randint(1, min(bs, mx_roll))
            else:
                roll_seed = 1
            latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], roll_seed, dims=0), latent_splits[1]), 0)
        return latent_feat_mismatch, latent_feat_relevant

    # TODO: debug this 
    def preprocess_latent_feat(self, latent_feat, mx_roll=1): # B x T x D
        if mx_roll > 1:
                roll_seed = np.random.randint(1, mx_roll)
        else:
            roll_seed = 1
        latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=1)
        bs = int(latent_feat.size(1)/2)
        latent_splits = torch.split(latent_feat, bs, 1)

        if mx_roll > 1:
            roll_seed = np.random.randint(1, min(mx_roll, bs))
        else:
            roll_seed = 1
        latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], roll_seed, dims=1), latent_splits[1]), 1)
        return latent_feat_mismatch, latent_feat_relevant
    
    def on_epoch_start(self,):
        self.trainer.train_dataloader.dataset.datasets.base_seed = np.random.randint(1000000)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        vid = batch['img'] # B x T x C x H x W 
        input_desc = batch['raw_desc'] # B
        sampleT = batch['sampleT']  # B x T 
        inversions = batch['inversion'] # B, T x n_layers x D
        
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
        
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=bs)
        B, n_frames, n_channels, dim = inversions.shape
        
        inversions = inversions.permute(1, 0, 2, 3)
        inversions = inversions.contiguous().reshape(T * B, n_channels, dim) # T * B x n_layers x D

        # Separating the content and the style of the inversions
        proj_content = self.content_proj(inversions) # B * T x C x content_dim
        proj_style = self.style_proj(inversions) # B * T x C x style_dim

        #TODO: Add max over style


        # TODO: Need to get attributes from loader (what format?)
        # Currently assuming one-hot vector for attributes, in attr variable
        # TODO: Make classifier one layer (no nonlinearities)
        # TODO: InfoMax on the style vector (like attribute removal paper)
        #classifier_probs = self.classifier(proj_style)
        #classifier_loss = F.cross_entropy(classifier_probs, attr)

        #TODO: InfoMin on the content vector

       # TODO: Do we apply the delta_inversion_weight to the results of the mapping network, or 
        # after the projection back into 512 dims?

        #TODO: Second classifier on the reconstructed image

        txt_feat_proj = self.clip_proj(txt_feat)
        txt_feat_mismatch_proj = self.clip_proj(txt_feat_mismatch)


        # Feed the content through the HairCLIP mappping network
        adjusted_latent_offset = self.delta_inversion_weight * self.mapping_network(proj_content, txt_feat_proj)
        adjusted_mismatched_latent_offset = self.delta_inversion_weight * self.mapping_network(proj_content, txt_feat_mismatch_proj)

        adjusted_latent = inversions + self.final_proj(adjusted_latent_offset)
        adjusted_mismatched_latent = inversions + self.final_proj(adjusted_mismatched_latent_offset)
        
        mismatched_img = self.G(adjusted_mismatched_latent) #.reshape(bs * T, ch, height, width)
        reconstructed = self.G(adjusted_latent) #.reshape(bs * T, ch, height, width)

        reconstructed = nn.functional.interpolate(reconstructed, size=(256, 192), mode="bicubic", align_corners=False)
        mismatched_img = nn.functional.interpolate(mismatched_img, size=(256, 192), mode="bicubic", align_corners=False)
        
        reconstruction_loss = self.reconstruction_loss(reconstructed, video_sample_norm)
        directional_clip_loss = self.directional_clip_loss(video_sample_norm, txt_feat, mismatched_img, txt_feat_mismatch, self.global_step)
        latent_loss = self.l2_latent_loss(inversions, adjusted_latent)
        latent_loss += torch.maximum(self.l2_latent_loss(inversions, adjusted_mismatched_latent) - self.l2_latent_eps, torch.zeros(1).to(inversions.device)[0])
        vgg_loss = self.criterionVGG(reconstructed, video_sample_norm)
        
        self.log("train/vgg_loss", vgg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train/rl_loss", self.rec_loss_lambda * reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/clip_loss",  self.clip_loss_lambda *  directional_clip_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/l2_latent_loss", self.l2_latent_lambda * latent_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        total_loss = self.rec_loss_lambda * reconstruction_loss + self.clip_loss_lambda * directional_clip_loss + self.l2_latent_lambda * latent_loss + vgg_loss
        
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
        
        B, n_frames, n_channels, dim = inversions.shape
        inversions = inversions.permute(1, 0, 2, 3)
        inversions = inversions.contiguous().reshape(B * n_frames, n_channels, dim)

        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # T*B x C x H x W // range [-1,1]
        ret['real_images'] = video_sample_norm
        
        txt_feat = self.get_text_embedding(input_desc) # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames, 1, 1) # T x B x D
        txt_feat = txt_feat.view(bs * n_frames, -1) # T * B x D
        
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=1)
        adjusted_latent = inversions + self.delta_inversion_weight * self.mapping_network(inversions, txt_feat)
        adjusted_mismatched_latent = inversions + self.delta_inversion_weight * self.mapping_network(inversions, txt_feat_mismatch)

        mismatched_img = self.G(adjusted_mismatched_latent) #.reshape(T ( B), ch, height, width)
        reconstructed = self.G(adjusted_latent) #.reshape(T * B, ch, height, width)

        ret['reconstruction'] = reconstructed
        ret['mismatched_text'] = mismatched_img
        return ret
             
        
           
        
        
     
        
                
        
    
        
        
        

        


