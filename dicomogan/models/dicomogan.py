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
from dicomogan.losses.loss_lib import GANLoss, VGGLoss, HybridOptim
import os
from PIL import Image
import random
import clip
from PIL import Image
import math

class DiCoMOGAN(pl.LightningModule):
    def __init__(self,
                    vae_cond_dim,
                    beta,
                    ODE_func_config, 
                    video_ecnoder_config,
                    video_decoder_config,
                    text_encoder_config,
                    discriminator_config,
                    generator_config,
                    mapping_config,
                    lambda_unsup,
                    lambda_vgg,
                    lambda_G,
                    lambda_bvae,
                    lambda_D,
                    scheduler_config = None,
                    custom_loggers = None,
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
        self.lambda_unsup = lambda_unsup
        self.lambda_vgg = lambda_vgg
        self.lambda_G = lambda_G
        self.lambda_bvae = lambda_bvae
        self.lambda_D = lambda_D

        # initialize model
        self.func = instantiate_from_config(ODE_func_config) 
        self.bVAE_enc = instantiate_from_config(video_ecnoder_config, odefunc=self.func) 
        self.bVAE_dec = instantiate_from_config(video_decoder_config)
        self.text_enc = instantiate_from_config(text_encoder_config)
        self.G = instantiate_from_config(generator_config)
        self.mapping = instantiate_from_config(mapping_config) 
        self.D = instantiate_from_config(discriminator_config)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32") # TODO: if not gonna use clip, then load all embeds before training. Save 3g memeory
        self.clip_model = self.clip_model 
        self.clip_model.requires_grad_(False)

        # loss
        self.criterionGAN = GANLoss(use_lsgan=True, target_real_label=1.0)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = VGGLoss()
        self.criterionUnsupFactor = torch.nn.MSELoss()

    
    def preprocess_feat(self, latent_feat):
            latent_feat = latent_feat.clone()
            bs = int(latent_feat.size(0)/2)
            latent_feat_mismatch = torch.roll(latent_feat, 1, 0)
            latent_splits = torch.split(latent_feat, bs, 0)
            latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], -1, 0), latent_splits[1]), 0)
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
        text = clip.tokenize(texts).to(self.device)
        return self.clip_model.encode_text(text).float()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        vid = batch['img'] # B x T x ch x H x W 
        input_desc = batch['raw_desc'] 
        sampleT = batch['sampleT'] # B x T 

        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        n_frames = sampleT.shape[0]
        bs, T, ch, height, width = vid.size()
        ts = (sampleT) * 0.01
        ts = ts - ts[0]
        
        video_sample = vid # B x T x C x H x W 
        video_sample = video_sample.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = video_sample.contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        
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
                               
        if optimizer_idx == 0:
            total_loss = 0
            vgg_loss = 0
            beta_vae_loss = 0
            G_loss = 0
            unsup_loss = 0

            # train vae and generator            
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
            self.log("train/beta_vae_loss", beta_vae_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)


            #  generator loss
            zsplits = torch.split(z_vid, int((bs * n_frames)/2), 0)
            z_rel = torch.cat((torch.roll(zsplits[0], -1, 0), zsplits[1]), 0)

            # calculate unsuper loss
            def unsupervised_loss(fake_sample, z_relative):
                fake_rs = nn.functional.interpolate(fake_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
                fake_sample_rs = fake_rs.view(n_frames, bs, ch, int(height*0.5), int(width*0.5))
                fake_sample_rs = fake_sample_rs.permute(1,0,2,3,4)
                zs_vid_fake, zd_vid_fake, mu_logvar_fake_s, mu_logvar_fake_d = self.bVAE_enc((fake_sample_rs+1)*0.5, ts)
                z_vid_fake = torch.cat((zs_vid_fake, zd_vid_fake), 1)
                return self.criterionUnsupFactor(z_vid_fake, z_relative)

            def GAN_VGG_Unsup_loss(video, text_feat, latnt, z_relative):
                fake = self.G(video, text_feat, latnt) 
                fake_logit = self.D(fake, text_feat, latnt)
                gan_loss = self.criterionGAN(fake_logit, True)
                vgg_loss = self.criterionVGG(fake, video)
                unsup_loss = unsupervised_loss(fake, z_relative)
                return gan_loss, vgg_loss, unsup_loss


            latentw = self.mapping(z_vid[:,self.vae_cond_dim:])
            _,txt_feat_relevant = self.preprocess_feat(txt_feat)
            _,latentw_relevant = self.preprocess_feat(latentw)

            gan_loss1, vgg_loss1, unsup_loss1 = GAN_VGG_Unsup_loss(video_sample_norm, txt_feat_relevant, latentw, z_rel)
            gan_loss2, vgg_loss2, unsup_loss2 = GAN_VGG_Unsup_loss(video_sample_norm, txt_feat, latentw_relevant, z_rel)
            gan_loss3, vgg_loss3, unsup_loss3 = GAN_VGG_Unsup_loss(video_sample_norm, txt_feat_relevant, latentw_relevant, z_rel)

            G_loss = (gan_loss1 + gan_loss2 + gan_loss3) / 3
            vgg_loss = (vgg_loss1 + vgg_loss2 + vgg_loss3) / 3
            unsup_loss = (unsup_loss1 + unsup_loss2 + unsup_loss3) / 3

            self.log("train/G_loss", G_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("train/vgg_loss", vgg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("train/unsup_loss", unsup_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            total_loss = self.lambda_bvae * beta_vae_loss + self.lambda_G * G_loss + self.lambda_vgg * vgg_loss + self.lambda_unsup * unsup_loss
            self.log("train/generator_loss", total_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return total_loss
            
        else:
            D_loss = 0
            # discriminator training
            def Disc_loss(vid, txt_feat, latent, real):
                logit = self.D(vid, txt_feat, latent)
                return self.criterionGAN(logit, real)
            
            # prepare latents
            latentw = self.mapping(z_vid[:,self.vae_cond_dim:])
            txt_feat_mismatch, txt_feat_relevant = self.preprocess_feat(txt_feat)
            latentw_mismatch, latentw_relevant = self.preprocess_feat(latentw)

            # real image with real latent)
            D_loss += Disc_loss(video_sample_norm, txt_feat, latentw, True)

            # real image with mismatching text
            D_loss += Disc_loss(video_sample_norm, txt_feat_mismatch, latentw, False)

            # real image with mismatching vae
            D_loss += Disc_loss(video_sample_norm, txt_feat, latentw_mismatch, False)

            # real image with mismatching text and vae
            D_loss += Disc_loss(video_sample_norm, txt_feat_mismatch, latentw_mismatch, False)

            # synthesized image with semantically relevant text
            fake = self.G(video_sample_norm, txt_feat_relevant, latentw)
            D_loss += Disc_loss(fake.detach(), txt_feat_relevant, latentw, False)

            # synthesized image with semantically relevant vae
            fake = self.G(video_sample_norm, txt_feat, latentw_relevant)
            D_loss += Disc_loss(fake.detach(), txt_feat, latentw_relevant, False)

            # synthesized image with semantically relevant text and vae
            fake = self.G(video_sample_norm, txt_feat_relevant, latentw_relevant)
            D_loss += Disc_loss(fake.detach(), txt_feat_relevant, latentw_relevant, False)
            
            D_loss /= 7 # normalize according to the number of samples
            self.log("train/D_loss", D_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            return self.lambda_D * D_loss

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
        
        ret['image'] = vid_rs_full * 2 - 1
        ret['x_recon_vae_text'] = self.bVAE_dec(torch.cat((zT,z_vid[:, self.vae_cond_dim:]), 1)) * 2 - 1 # T*B x C x H x W
        ret['x_recon_vae'] = self.bVAE_dec(z_vid) * 2 - 1 # T*B x C x H x W

        # generate with mathching text
        latentw = self.mapping(z_vid[:,self.vae_cond_dim:])
        mismatch_txt_feat, txt_feat_relevant = self.preprocess_feat(txt_feat)
        mismatche_latentw, latentw_relevant = self.preprocess_feat(latentw)

        ret['x_recon_gan']  = self.G(video_sample * 2 - 1, txt_feat, latentw)
        ret['x_mismatch_text']  = self.G(video_sample * 2 - 1, mismatch_txt_feat, latentw)
        ret['x_mismatch_wcont']  = self.G(video_sample * 2 - 1, txt_feat, mismatche_latentw)

        return ret



    # TODO: Check betas and epsilons of the optimizers
    def configure_optimizers(self):
        lr = self.learning_rate
        vae_params = list(self.bVAE_enc.parameters())+\
                 list(self.bVAE_dec.parameters())+\
                 list(self.text_enc.parameters())

        G_params = list(self.G.parameters())

        m_params = list(self.mapping.parameters())

        dis_params = list(self.D.parameters())

        opt_vae = torch.optim.Adam(vae_params,
                                  lr=lr * 5, 
                                  betas=(0.9, 0.999))
        
        opt_g = torch.optim.Adam(G_params,
                                  lr=lr, 
                                  betas=(0.5, 0.999))
        
        opt_m = torch.optim.Adam(m_params,
                                  lr=lr / 100, 
                                  betas=(0.5, 0.999))

        opt_ae = HybridOptim([opt_vae, opt_g, opt_m])
        
        
        ae_ret = {"optimizer": opt_ae, "frequency": 1}

        ae_scheduler = None
        if self.scheduler_config is not None:
            # TODO: scheduler not implemented
            ae_scheduler = LambdaLR(opt_ae, lr_lambda=instantiate_from_config(self.scheduler_config).schedule, verbose=False)
            ae_ret['lr_scheduler'] = {
                        'scheduler': ae_scheduler, 
                        'interval': 'step'}

        opt_disc = torch.optim.Adam(dis_params,
                                lr=lr, betas=(0.5, 0.99)) 
        
        disc_ret = {"optimizer": opt_disc, "frequency": 1}
        if self.scheduler_config is not None:
            # TODO: scheduler not implemented
            dis_scheduler = LambdaLR(opt_disc, lr_lambda=instantiate_from_config(self.scheduler_config).schedule, verbose=False)
            disc_ret['lr_scheduler'] = {
                    'scheduler': dis_scheduler, 
                    'interval': 'step'}

        if self.n_critic < 0:
            ae_ret['frequency'] = -self.n_critic
        else:
            disc_ret['frequency'] = self.n_critic
        
        return [ae_ret, disc_ret]