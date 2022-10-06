import pytorch_lightning as pl
from experiments_utils import *
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dicomogan.losses.clip_loss import CLIPLoss
from dicomogan.losses.loss_lib import GANLoss, VGGLoss, HybridOptim
from dicomogan.metrics.inception_score import inception_score
from dicomogan.metrics.extra_metrics import clip_score
from dicomogan.metrics.frechet_video_distance.frechet_video_distance import frechet_video_distance
import os
from PIL import Image
import random
import math

class DiCoMOGANCLIP(pl.LightningModule):
    def __init__(self,
                    vae_cond_dim,
                    beta,
                    ODE_func_config, 
                    video_ecnoder_config,
                    video_decoder_config,
                    text_encoder_config,
                    style_mapper_config,
                    stylegan_gen_config,
                    mapping_config,
                    lambda_vgg,
                    lambda_G,
                    lambda_bvae,
                    rec_loss_lambda,
                    l2_latent_lambda,
                    clip_loss_lambda,
                    consistency_lambda,
                    delta_inversion_weight,
                    l2_latent_eps,
                    scheduler_config = None,
                    custom_loggers = None,
                    tgt_text = None,
                    n_critic = 1,
                    ):
        super().__init__()
        self.clip_img_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224), 
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.vae_cond_dim = vae_cond_dim
        self.beta = beta
        self.scheduler_config = scheduler_config
        self.n_critic = n_critic

        # Loss weights
        self.lambda_vgg = lambda_vgg
        self.lambda_G = lambda_G
        self.lambda_bvae = lambda_bvae
        self.rec_loss_lambda = rec_loss_lambda
        self.l2_latent_lambda = l2_latent_lambda
        self.clip_loss_lambda = clip_loss_lambda
        self.consistency_lambda = consistency_lambda
        self.delta_inversion_weight = delta_inversion_weight
        self.l2_latent_eps = l2_latent_eps

        # for later
        # self.lambda_unsup = lambda_unsup
        # self.lambda_D = lambda_D

        # initialize model
        self.func = instantiate_from_config(ODE_func_config) 
        self.bVAE_enc = instantiate_from_config(video_ecnoder_config, odefunc=self.func) 
        self.bVAE_dec = instantiate_from_config(video_decoder_config)
        self.text_enc = instantiate_from_config(text_encoder_config)
        self.mapping = instantiate_from_config(mapping_config) 
        self.style_mapper = instantiate_from_config(style_mapper_config)
        self.stylegan_G = instantiate_from_config(stylegan_gen_config)
        self.requires_grad(self.stylegan_G, False)
        self.stylegan_G.eval() # TODO: rm the eval 

        # loss
        self.criterionVGG = VGGLoss()
        self.rec_loss = nn.MSELoss()
        self.l2_latent_loss = nn.MSELoss()
        self.clip_loss = CLIPLoss()

        self.tgt_text_embed = None
        if tgt_text is not None:
            self.tgt_text_embed = self.clip_encode_text([tgt_text]) # 1 x 512

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def preprocess_text_feat(self, latent_feat, mx_roll=2):
        bs = int(latent_feat.size(0)/2)
        if self.tgt_text_embed is not None:
            self.tgt_text_embed = self.tgt_text_embed.to(latent_feat.device)
            latent_feat_mismatch = self.tgt_text_embed.repeat(latent_feat.size(0), 1)
            latent_splits = torch.split(latent_feat, bs, 0)
            latent_feat_relevant = torch.cat((self.tgt_text_embed.repeat(bs, 1), latent_splits[1]), 0)
        else:
            roll_seed = np.random.randint(1, mx_roll)
            latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=0)
            latent_splits = torch.split(latent_feat, bs, 0)

            roll_seed = np.random.randint(1, min(bs, mx_roll))
            latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], roll_seed, dims=0), latent_splits[1]), 0)
        return latent_feat_mismatch, latent_feat_relevant

    # TODO: debug this 
    def preprocess_latent_feat(self, latent_feat, mx_roll=2): # B x T x D
        roll_seed = np.random.randint(1, mx_roll)
        latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=1)
        bs = int(latent_feat.size(1)/2)
        latent_splits = torch.split(latent_feat, bs, 1)

        roll_seed = np.random.randint(1, min(mx_roll, bs))
        latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], roll_seed, dims=1), latent_splits[1]), 1)
        return latent_feat_mismatch, latent_feat_relevant

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
            #recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False)

        elif distribution == 'gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False)
        else:
            recon_loss = None

        return recon_loss/batch_size


    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        latent_dim = mu.size(1)

        latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()
        return total_kl

    def reparametrize(self, mu, logvar):
            std = logvar.div(2).exp()
            eps = std.data.new(std.size()).normal_()
            return mu + std*eps

    def clip_encode_text(self, texts):
        text = self.clip_loss.tokenize(texts).to(self.device)
        return self.clip_loss.encode_text(text).float()
    
    def on_train_epoch_start(self,):
            self.trainer.train_dataloader.dataset.datasets.reset()

    def training_step(self, batch, batch_idx):
        input_desc = batch['raw_desc'] # B
        sampleT = batch['sampleT']  # B x T 
        

        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        ts = (sampleT) * 0.01
        ts = ts - ts[0]

        # videos reshape
        vid_bf = batch['real_img'] # B x T x C x H x W 
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = vid_tf.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        
        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        inversions = inversions_tf.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D
        inversions.requires_grad = False

        # downsample res for vae TODO: experiment with downsampling the resolution much more
        vid_rs = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True) # T*B x C x H//2 x W//2 
        vid_rs_tf = vid_rs.view(n_frames, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs_bf = vid_rs_tf.permute(1,0,2,3,4) # B x T x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(n_frames,1,1)
        txt_feat = txt_feat_tf.contiguous().view(bs * n_frames, -1)  # T*B x D
        txt_feat.requires_grad = False

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs_bf, ts)
        z_vid = torch.cat((zs, zd), 1) # T*B x D 
                               
        total_loss = 0
        vgg_loss = 0 
        beta_vae_loss = 0
        G_loss = 0

        # Train vae and generator 
                   
        # vae encode text
        muT, logvarT = self.text_enc(txt_feat)
        zT = self.reparametrize(muT, logvarT) # T*B x D 

        x_reconT = self.bVAE_dec(torch.cat((zT,z_vid[:, self.vae_cond_dim:]), 1)) # T*B x C x H x W
        x_recon = self.bVAE_dec(z_vid) # T*B x C x H x W

        recon_loss = self.reconstruction_loss(vid_rs, x_recon, 'bernoulli')
        recon_lossT = self.reconstruction_loss(vid_rs, x_reconT, 'bernoulli')
        self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/recon_lossT", recon_lossT, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        kl_loss_d = self.beta * self.kl_divergence(*mu_logvar_d)
        kl_loss_s = self.beta * self.kl_divergence(*mu_logvar_s)
        kl_loss = kl_loss_s +  kl_loss_d
        self.log("train/kl_loss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        beta_vae_loss = 0.5 * (recon_loss + recon_lossT) +  kl_loss
        self.log("train/beta_vae_loss", self.lambda_bvae * beta_vae_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)


        # Generator loss
        latentw = self.mapping(z_vid[:,self.vae_cond_dim:]) # T*B x D

        # roll video-wise
        # reshaped_latentw = latentw.view(T, bs, -1).permute(1, 0, 2).contiguous()
        # _,latentw_relevant = self.preprocess_latent_feat(reshaped_latentw) 
        # latentw_relevant = latentw_relevant.permute(1, 0, 2).contiguous().view(T*bs, -1) # T*B x D1
        
        # roll batch-wise
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=bs) # T*B x D2
        
        frame_rep = torch.cat((latentw, txt_feat), -1) # T*B x D1+D2
        frame_rep_txt_mismatched = torch.cat((latentw, txt_feat_mismatch), -1) # T*B x D1+D2

        # predict latents delta
        src_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        src_inversion_tf = src_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, n_channels, dim)
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep)
        w_latents_txt_mismatched = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep_txt_mismatched)

        reconstruction = self.stylegan_G(w_latents) # T*B x 3 x H x W
        imgs_txt_mismatched = self.stylegan_G(w_latents_txt_mismatched) # T*B x 3 x H x W

        reconstruction_inp_res = nn.functional.interpolate(reconstruction, size=(256, 192), mode="bicubic", align_corners=False)
        imgs_txt_mismatched_inp_res = nn.functional.interpolate(imgs_txt_mismatched, size=(256, 192), mode="bicubic", align_corners=False)

        reconstruction_loss = self.rec_loss(reconstruction_inp_res, video_sample_norm)
        latent_loss = self.l2_latent_loss(inversions, w_latents)
        latent_loss += torch.maximum(self.l2_latent_loss(inversions, w_latents_txt_mismatched) - self.l2_latent_eps, torch.zeros(1).to(inversions.device)[0])
        vgg_loss = self.criterionVGG(reconstruction_inp_res, video_sample_norm)
        
        
        # video based losses
        txt_feat_bf = txt_feat_tf.permute(1, 0, 2).contiguous() # B x T x D

        txt_feat_mismatch_tf = txt_feat_mismatch.contiguous().reshape(T, bs, txt_feat_mismatch.shape[1])
        txt_feat_mismatch_bf = txt_feat_mismatch_tf.permute(1, 0, 2).contiguous()  # B x T x D

        vid_norm_bf = vid_bf * 2 - 1

        reconstruction_tf = reconstruction.contiguous().reshape(T, bs, reconstruction.shape[1], reconstruction.shape[2], reconstruction.shape[3])
        reconstruction_bf = reconstruction_tf.permute(1, 0, 2, 3, 4).contiguous()  # B x T x C x H W 

        imgs_txt_mismatched_tf = imgs_txt_mismatched.contiguous().reshape(T, bs, imgs_txt_mismatched.shape[1], imgs_txt_mismatched.shape[2], imgs_txt_mismatched.shape[3])
        imgs_txt_mismatched_bf = imgs_txt_mismatched_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W 

        imgs_txt_mismatched_inp_res_tf = imgs_txt_mismatched_inp_res.contiguous().reshape(T, bs, imgs_txt_mismatched_inp_res.shape[1], imgs_txt_mismatched_inp_res.shape[2], imgs_txt_mismatched_inp_res.shape[3])
        imgs_txt_mismatched_inp_res_bf = imgs_txt_mismatched_inp_res_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W 

        # directional loss
        directional_clip_loss = self.clip_loss.directional_loss(vid_norm_bf, txt_feat_bf, imgs_txt_mismatched_inp_res_bf, txt_feat_mismatch_bf, self.global_step, video=True)

        # consistency loss
        consistency_loss = 0
        consistency_loss = self.clip_loss.consistency_loss(reconstruction_bf)
        consistency_loss += self.clip_loss.consistency_loss(imgs_txt_mismatched_bf)

        self.log("train/consistency_loss", consistency_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train/vgg_loss", vgg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train/rl_loss", self.rec_loss_lambda * reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/clip_loss",  self.clip_loss_lambda *  directional_clip_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/l2_latent_loss", self.l2_latent_lambda * latent_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        G_loss = self.rec_loss_lambda * reconstruction_loss + self.clip_loss_lambda * directional_clip_loss + self.l2_latent_lambda * latent_loss
        self.log("train/G_loss", G_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        total_loss = self.lambda_bvae * beta_vae_loss + self.lambda_G * G_loss + self.lambda_vgg * vgg_loss + self.consistency_lambda * consistency_loss
        self.log("train/total_loss", total_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return total_loss



    def validation_step(self, batch, batch_idx):
        input_desc = batch['raw_desc'] # B
        sampleT = batch['sampleT'] # B x T
        vid_bf = batch['real_img'] # B x T x C x H x W
        inversions_bf = batch['inversion'] # B x T x n_layers x D

        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        ts = (sampleT) * 0.01
        ts = ts - ts[0]

        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = vid_tf.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc

        n_channels, dim = inversions_bf.shape[2], inversions_bf.shape[3]

        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        inversions = inversions_tf.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D
        inversions.requires_grad = False

        vid_rs = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True) # T*B x C x H//2 x W//2 
        vid_rs_tf = vid_rs.view(n_frames, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs_bf = vid_rs_tf.permute(1,0,2,3,4) # B x T x C x H//2 x W//2

        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(n_frames,1,1)
        txt_feat = txt_feat_tf.contiguous().view(bs * n_frames, -1)  # T*B x D
        txt_feat.requires_grad = False

        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs_bf, ts)
        z_vid = torch.cat((zs, zd), 1) # T*B x D 

        muT, logvarT = self.text_enc(txt_feat)
        zT = self.reparametrize(muT, logvarT) # T*B x D 

        x_reconT = self.bVAE_dec(torch.cat((zT,z_vid[:, self.vae_cond_dim:]), 1)) # T*B x C x H x W
        x_recon = self.bVAE_dec(z_vid) # T*B x C x H x W

        latentw = self.mapping(z_vid[:,self.vae_cond_dim:]) # T*B x D

        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=bs) # T*B x D2
        
        frame_rep = torch.cat((latentw, txt_feat), -1) # T*B x D1+D2
        frame_rep_txt_mismatched = torch.cat((latentw, txt_feat_mismatch), -1) # T*B x D1+D2

        # predict latents delta
        src_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        src_inversion_tf = src_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, n_channels, dim)
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep)
        w_latents_txt_mismatched = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep_txt_mismatched)

        reconstruction = self.stylegan_G(w_latents) # T*B x 3 x H x W
        imgs_txt_mismatched = self.stylegan_G(w_latents_txt_mismatched) # T*B x 3 x H x W

        reconstruction_inp_res = nn.functional.interpolate(reconstruction, size=(256, 192), mode="bicubic", align_corners=False)
        imgs_txt_mismatched_inp_res = nn.functional.interpolate(imgs_txt_mismatched, size=(256, 192), mode="bicubic", align_corners=False)

        # video based losses
        txt_feat_bf = txt_feat_tf.permute(1, 0, 2).contiguous() # B x T x D

        txt_feat_mismatch_tf = txt_feat_mismatch.contiguous().reshape(T, bs, txt_feat_mismatch.shape[1])
        txt_feat_mismatch_bf = txt_feat_mismatch_tf.permute(1, 0, 2).contiguous()  # B x T x D

        vid_norm_bf = vid_bf * 2 - 1

        reconstruction_tf = reconstruction.contiguous().reshape(T, bs, reconstruction.shape[1], reconstruction.shape[2], reconstruction.shape[3])
        reconstruction_bf = reconstruction_tf.permute(1, 0, 2, 3, 4).contiguous()  # B x T x C x H W 

        imgs_txt_mismatched_tf = imgs_txt_mismatched.contiguous().reshape(T, bs, imgs_txt_mismatched.shape[1], imgs_txt_mismatched.shape[2], imgs_txt_mismatched.shape[3])
        imgs_txt_mismatched_bf = imgs_txt_mismatched_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W 

        imgs_txt_mismatched_inp_res_tf = imgs_txt_mismatched_inp_res.contiguous().reshape(T, bs, imgs_txt_mismatched_inp_res.shape[1], imgs_txt_mismatched_inp_res.shape[2], imgs_txt_mismatched_inp_res.shape[3])
        imgs_txt_mismatched_inp_res_bf = imgs_txt_mismatched_inp_res_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W 

        is_metric_recon = inception_score(reconstruction_inp_res)
        is_metric_mismatched = inception_score(imgs_txt_mismatched_inp_res)

        is_metric_original = inception_score(video_sample)

        clip_metric_recon = clip_score(self.clip_loss, reconstruction_inp_res, txt_feat, fake_img=True)
        clip_metric_mismatched = clip_score(self.clip_loss, imgs_txt_mismatched_inp_res, txt_feat_mismatch, fake_img=True)
        clip_metric_original = clip_score(self.clip_loss, video_sample, txt_feat, fake_img=False)

        #reconstruction_inp_res = reconstruction_inp_res.cpu()
        #imgs_txt_mismatched_inp_res = imgs_txt_mismatched_inp_res.cpu()
        txt_feat = txt_feat.cpu()
        txt_feat_mismatch = txt_feat_mismatch.cpu()


        self.log("val/inception_recon", is_metric_recon[0], prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("val/inception_mismatched", is_metric_mismatched[0], prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("val/inception_original", is_metric_original[0], prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("val/clip_recon", clip_metric_recon, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("val/clip_mismatched", clip_metric_mismatched, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("val/clip_original", clip_metric_original, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return_dict = dict()
        return_dict['is_recon'] = is_metric_recon[0]
        return_dict['is_mismatched'] = is_metric_mismatched[0]
        return_dict['is_original'] = is_metric_original[0]
        return_dict['clip_recon'] = clip_metric_recon
        return_dict['clip_mismatched'] = clip_metric_mismatched
        return_dict['clip_original'] = clip_metric_original
        return_dict['original_imgs'] = video_sample.reshape(bs, n_frames, ch, height, width)
        return_dict['recon_imgs'] = reconstruction_inp_res.reshape(bs, n_frames, video_sample.shape[1], video_sample.shape[2], video_sample.shape[3])
        return_dict['mismatched_imgs'] = imgs_txt_mismatched_inp_res.reshape(bs, n_frames, video_sample.shape[1], video_sample.shape[2], video_sample.shape[3])

        return return_dict

    def validation_epoch_end(self, validation_step_outputs):
        total_is_recon = []
        total_is_mismatched = []
        total_is_original = []
        total_clip_recon = []
        total_clip_mismatched = []
        total_clip_original = []

        all_originals = []
        recon_imgs = []
        mismatched_imgs = []

        for output in validation_step_outputs:
            total_is_recon.append(output['is_recon'])
            total_is_mismatched.append(output['is_mismatched'])
            total_is_original.append(output['is_original'])
            total_clip_recon.append(output['clip_recon'])
            total_clip_mismatched.append(output['clip_mismatched'])
            total_clip_original.append(output['clip_original'])
            all_originals.append(output['original_imgs'])
            recon_imgs.append(output['recon_imgs'])
            mismatched_imgs.append(output['mismatched_imgs'])


        all_originals = torch.stack(all_originals, dim=0)
        recon_imgs = torch.stack(recon_imgs, dim=0)
        mismatched_imgs = torch.stack(mismatched_imgs, dim=0)

        num_batches, batch_size, n_frames, ch, height, width = all_originals.shape
        all_originals = all_originals.reshape(num_batches * batch_size, n_frames, ch, height, width)
        recon_imgs = recon_imgs.reshape(num_batches * batch_size, n_frames, ch, height, width)
        mismatched_imgs = mismatched_imgs.reshape(num_batches * batch_size, n_frames, ch, height, width)

        model_path = "/kuacc/users/abond19/VideoEditing/dicomogan/dicomogan/metrics/frechet_video_distance/pytorch_i3d_model/models/rgb_imagenet.pt"

        #print(f"Originals shape: {all_originals.shape}")
        #print(f"Reconstruction shape: {recon_imgs.shape}")
        #print(f"Mismatched shape: {mismatched_imgs.shape}")

        recon_fvd = frechet_video_distance(all_originals, recon_imgs, model_path, self.device)
        mismatched_fvd = frechet_video_distance(all_originals, mismatched_imgs, model_path, self.device)

        self.log("val/mean_inception_recon", np.array(total_is_recon).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_inception_mismatched", np.array(total_is_mismatched).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_inception_original", np.array(total_is_original).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_clip_recon", np.array(total_clip_recon).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_clip_mismatched", np.array(total_clip_mismatched).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mean_clip_original", np.array(total_clip_original).mean(), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/recon_fvd", recon_fvd, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mismatched_fvd", mismatched_fvd, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def log_images(self, batch, split):
        """
        return a dictionary of tensors in the range [-1, 1]
        """
        ret = dict()

        vid = batch['real_img'] # B x T x ch x H x W -- range [0, 1]
        inverted_vid_bf = batch['inverted_img'] # B x T x ch x H x W -- range [0, 1]
        input_desc = batch['raw_desc'] 
        sampleT = batch['sampleT'] # B x T 
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        bs, T, ch, height, width = vid.size()
        ts = (sampleT)*0.01
        ts = ts - ts[0] 
        
        inverted_vid_tf = inverted_vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        inverted_vid = inverted_vid_tf.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W 
        inverted_vid_norm = inverted_vid * 2 - 1 # range [-1, 1] to pass to the generator and disc

        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W 
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        
        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        inversions = inversions_tf.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D

        # downsample res for vae
        vid_rs_full = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
        vid_rs = vid_rs_full.view(n_frames, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs = vid_rs.permute(1,0,2,3,4) #  B x T x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames,1,1)
        txt_feat = txt_feat.view(bs * n_frames, -1)  # T*B x D

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs, ts)
        z_vid = torch.cat((zs, zd), 1) # T*B x D 

        muT, logvarT = self.text_enc(txt_feat)
        zT = self.reparametrize(muT, logvarT) # T*B x D 
        
        # generate with mathching text
        latentw = self.mapping(z_vid[:,self.vae_cond_dim:])
        
        # roll video-wise
        # reshaped_latentw = latentw.view(T, bs, -1).permute(1, 0, 2).contiguous()
        # mismatche_latentw, _ = self.preprocess_latent_feat(reshaped_latentw) # B x T x D
        # mismatche_latentw = mismatche_latentw.permute(1, 0, 2).contiguous().view(T*bs, -1)

        # roll batch-wise
        mismatch_txt_feat, _ = self.preprocess_text_feat(txt_feat, mx_roll=2)

        frame_rep = torch.cat((latentw, txt_feat), -1) # T*B x D1+D2
        frame_rep_txt_mismatched = torch.cat((latentw, mismatch_txt_feat), -1) # T*B x D1+D2

        # predict latents delta
        src_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        src_inversion_tf = src_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, n_channels, dim)
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep)
        w_latents_txt_mismatched = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, frame_rep_txt_mismatched)

        ret['real_image'] = video_sample_norm 
        ret['inverted_image'] = inverted_vid_norm 
        ret['real_image_caption'] = '\n'.join([f"Col_{i}: {el}" for i, el in enumerate(batch['raw_desc'])])
        
        ret['x_recon_vae_text'] = self.bVAE_dec(torch.cat((zT,z_vid[:, self.vae_cond_dim:]), 1)) * 2 - 1 # T*B x C x H x W
        ret['x_recon_vae'] = self.bVAE_dec(z_vid) * 2 - 1 # T*B x C x H x W
        ret['vae_mismatched_video_style'] = self.bVAE_dec(torch.cat((torch.roll(z_vid[:, :self.vae_cond_dim], 1, dims=1), z_vid[:, self.vae_cond_dim:]), 1)) * 2 - 1 # T*B x C x H x W
        recon_gan = self.stylegan_G(w_latents)
        mismatched_gan = self.stylegan_G(w_latents_txt_mismatched)
        
        ret['x_recon_gan']  = recon_gan
        ret['x_mismatch_text']  = mismatched_gan

        if split == 'val':
             # TODO: create gif instead of frames
            gifs = []
            if not os.path.exists("./tmp/"):
                os.mkdir("./tmp/")
            
            self.log_gif(recon_gan, (3, 3, 3, 1024, 512), "recon")
            self.log_gif(mismatched_gan, (3, 3, 3, 1024, 512), "mismatched")
        return ret

    
    def log_gif(self, video, shape, type):
        #import imageio
        #with imageio.get_writer(f'{save_path}.gif', mode='I') as writer:
        #    for img in img_lst:
        #        writer.append_data(np.array(img))

        # Assuming that the current shape is T * B x C x H x W
        video_correct = video.contiguous().reshape(shape[1], shape[0], shape[2], shape[3], shape[4])
        video_correct = (((video_correct.permute(1, 0, 3, 4, 2).cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
        print(np.max(video_correct), np.min(video_correct))

        for i in range(shape[0]):
            current_video = video_correct[i]
            imgs = [Image.fromarray(img) for img in current_video]
            imgs[0].save("./tmp/temp.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
            wandb.log({f"val/gif_{type}": wandb.Video("./tmp/temp.gif", fps=4, format="gif")})


    def configure_optimizers(self):
        lr = self.learning_rate
        vae_params = list(self.bVAE_enc.parameters())+\
                 list(self.bVAE_dec.parameters())+\
                 list(self.text_enc.parameters())

        m_params = list(self.mapping.parameters())

        style_m_params = list(self.style_mapper.parameters())
        
        opt_vae = torch.optim.Adam(vae_params,
                                  lr=lr * 5, 
                                  betas=(0.9, 0.999))

        opt_sm = torch.optim.Adam(style_m_params, lr=lr, betas=(0.9, 0.99))

        opt_m = torch.optim.Adam(m_params,
                                  lr=lr / 100, 
                                  betas=(0.5, 0.999))

        opt_ae = HybridOptim([opt_vae, opt_m, opt_sm])
        
        
        ae_ret = {"optimizer": opt_ae, "frequency": 1}

        # ae_scheduler = None
        # if self.scheduler_config is not None:
        #     # TODO: scheduler not implemented
        #     ae_scheduler = LambdaLR(opt_ae, lr_lambda=instantiate_from_config(self.scheduler_config).schedule, verbose=False)
        #     ae_ret['lr_scheduler'] = {
        #                 'scheduler': ae_scheduler, 
        #                 'interval': 'step'}

        # if self.scheduler_config is not None:
        #     # TODO: scheduler not implemented
        #     dis_scheduler = LambdaLR(opt_disc, lr_lambda=instantiate_from_config(self.scheduler_config).schedule, verbose=False)
        #     disc_ret['lr_scheduler'] = {
        #             'scheduler': dis_scheduler, 
        #             'interval': 'step'}
        
        return ae_ret