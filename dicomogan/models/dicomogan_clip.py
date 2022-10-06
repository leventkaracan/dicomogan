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
from dicomogan.losses.clip_loss import CLIPLoss
from dicomogan.losses.loss_lib import GANLoss, VGGLoss, HybridOptim
import os
from PIL import Image
import random
from PIL import Image
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
                    ckpt_path=None,
                    ignore_keys=[],
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
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

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
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction="mean")
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
    
    def on_epoch_start(self,):
            self.trainer.train_dataloader.dataset.datasets.base_seed = np.random.randint(1000000)

    def training_step(self, batch, batch_idx):
        input_desc = batch['raw_desc'] # B
        sampleT = batch['sampleT']  # B x T 
        

        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        ts = (sampleT) * 0.01
        ts = ts - ts[0]

        # videos reshape
        vid = batch['img'] # B x T x C x H x W 
        bs, T, ch, height, width = vid.size()
        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        
        # inversions reshape
        inversions = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions.shape
        inversions = inversions.permute(1, 0, 2, 3)
        inversions = inversions.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D
        inversions.requires_grad = False

        # downsample res for vae TODO: experiment with downsampling the resolution much more
        vid_rs_full = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
        vid_rs = vid_rs_full.view(n_frames, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs = vid_rs.permute(1,0,2,3,4) # T x B x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames,1,1)
        txt_feat = txt_feat.contiguous().view(bs * n_frames, -1)  # T*B x D
        txt_feat.requires_grad = False

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs, ts)
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

        recon_loss = self.reconstruction_loss(vid_rs_full, x_recon, 'bernoulli')
        recon_lossT = self.reconstruction_loss(vid_rs_full, x_reconT, 'bernoulli')
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
        w_latents = inversions + self.delta_inversion_weight * self.style_mapper(inversions, frame_rep)
        w_latents_txt_mismatched = inversions + self.delta_inversion_weight * self.style_mapper(inversions, frame_rep_txt_mismatched)

        reconstruction = self.stylegan_G(w_latents) # T*B x 3 x H x W
        imgs_txt_mismatched = self.stylegan_G(w_latents_txt_mismatched) # T*B x 3 x H x W

        reconstruction_res = nn.functional.interpolate(reconstruction, size=(256, 192), mode="bicubic", align_corners=False)
        imgs_txt_mismatched_res = nn.functional.interpolate(imgs_txt_mismatched, size=(256, 192), mode="bicubic", align_corners=False)

        reconstruction_loss = self.rec_loss(reconstruction_res, video_sample_norm)
        directional_clip_loss = self.clip_loss.directional_loss(video_sample_norm, txt_feat, imgs_txt_mismatched_res, txt_feat_mismatch, self.global_step)
        latent_loss = self.l2_latent_loss(inversions, w_latents)
        latent_loss += torch.maximum(self.l2_latent_loss(inversions, w_latents_txt_mismatched) - self.l2_latent_eps, torch.zeros(1).to(inversions.device)[0])
        vgg_loss = self.criterionVGG(reconstruction_res, video_sample_norm)
        

        # consistency loss
        consistency_loss = 0
        # reconstruction = reconstruction.reshape(T, bs, reconstruction.shape[2], reconstruction.shape[3], reconstruction.shaep[4])
        # reconstruction = reconstruction.permute(1, 0, 2, 3, 4)

        # imgs_txt_mismatched = imgs_txt_mismatched.reshape(T, bs, imgs_txt_mismatched.shape[2], imgs_txt_mismatched.shape[3], imgs_txt_mismatched.shaep[4])
        # imgs_txt_mismatched = imgs_txt_mismatched.permute(1, 0, 2, 3, 4)

        # consistency_loss = self.consistency_loss(reconstruction)
        # consistency_loss += self.consistency_loss(imgs_txt_mismatched)

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

    def log_images(self, batch, split):
        """
        return a dictionary of tensors in the range [-1, 1]
        """
        ret = dict()

        vid = batch['img'] # B x T x ch x H x W -- range [0, 1]
        input_desc = batch['raw_desc'] 
        sampleT = batch['sampleT'] # B x T 
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        bs, T, ch, height, width = vid.size()
        ts = (sampleT)*0.01
        ts = ts - ts[0] 
        
        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W 
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        
        # inversions reshape
        inversions = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions.shape
        inversions = inversions.permute(1, 0, 2, 3)
        inversions = inversions.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D

        # downsample res for vae
        vid_rs_full = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
        vid_rs = vid_rs_full.view(n_frames, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs = vid_rs.permute(1,0,2,3,4) # T x B x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat = txt_feat.unsqueeze(0).repeat(n_frames,1,1)
        txt_feat = txt_feat.view(bs * n_frames, -1)  # T*B x D

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs, ts)
        z_vid = torch.cat((zs, zd), 1) # T*B x D 

        muT, logvarT = self.text_enc(txt_feat)
        zT = self.reparametrize(muT, logvarT) # T*B x D 
        
        ret['image'] = vid_rs_full * 2 - 1 # downsampled version
        ret['image_caption'] = '\n'.join([f"Col_{i}: {el}" for i, el in enumerate(batch['raw_desc'])])
        ret['x_recon_vae_text'] = self.bVAE_dec(torch.cat((zT,z_vid[:, self.vae_cond_dim:]), 1)) * 2 - 1 # T*B x C x H x W
        ret['x_recon_vae'] = self.bVAE_dec(z_vid) * 2 - 1 # T*B x C x H x W

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
        w_latents = inversions + self.delta_inversion_weight * self.style_mapper(inversions, frame_rep)
        w_latents_txt_mismatched = inversions + self.delta_inversion_weight * self.style_mapper(inversions, frame_rep_txt_mismatched)

        ret['x_recon_gan']  = self.stylegan_G(w_latents)
        ret['x_mismatch_text']  = self.stylegan_G(w_latents_txt_mismatched)

        return ret



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