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
from dicomogan.losses.lpips import LPIPS
from dicomogan.losses.loss_lib import GANLoss, VGGLoss, HybridOptim
from dicomogan.models.utils import ConstScheduler
import os
from PIL import Image
import random
from PIL import Image
import math
import wandb
import imageio
import torchvision
import uuid


class DiCoMOGANCLIP(pl.LightningModule):
    def __init__(self,
                    beta,
                    video_ecnoder_config,
                    style_mapper_config,
                    stylegan_gen_config,
                    #mapping_config,
                    lambda_vgg,
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
                    ckpt_path=None,
                    ignore_keys=[], 
                    video_length = 6,
                    frame_log_size=(256, 192)
                    ):
        super().__init__()
        self.beta = beta
        self.scheduler_config = scheduler_config
        self.n_critic = n_critic
        self.frame_log_size = frame_log_size
        self.video_length = video_length

        self.delta_inversion_weight = delta_inversion_weight
        self.l2_latent_eps = l2_latent_eps
        self.lambda_vgg = self.prepare_scheduler(lambda_vgg)
        self.l2_latent_lambda = self.prepare_scheduler(l2_latent_lambda)
        self.clip_loss_lambda = self.prepare_scheduler(clip_loss_lambda)
        self.rec_loss_lambda = self.prepare_scheduler(rec_loss_lambda)
        self.consistency_lambda = self.prepare_scheduler(consistency_lambda)

        # initialize model
        self.bVAE_enc = instantiate_from_config(video_ecnoder_config) 
        self.style_mapper = instantiate_from_config(style_mapper_config)
        self.stylegan_G = instantiate_from_config(stylegan_gen_config)
        self.requires_grad(self.stylegan_G, False)
        self.stylegan_G.eval() # TODO: rm the eval maybe?

        # loss
        self.criterionVGG = LPIPS().eval()
        self.rec_loss = nn.MSELoss()
        self.l2_latent_loss = nn.MSELoss()
        self.clip_loss = CLIPLoss()

        self.tgt_text_embed = None
        if tgt_text is not None:
            self.tgt_text_embed = self.clip_encode_text([tgt_text]) # 1 x 512
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def prepare_scheduler(self, val):
        if isinstance(val, float):
            return ConstScheduler(val)
        else:
            return instantiate_from_config(val)

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
        latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=1) # toll on the frame dims

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
        latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()
        return total_kl

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps

    def clip_encode_text(self, texts):
        texts = self.clip_loss.tokenize(texts).to(self.device)
        return self.clip_loss.encode_text(texts).float()
    
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

    def on_train_epoch_start(self,):
            self.trainer.train_dataloader.dataset.datasets.reset()


    def forward(self, vid_bf, sampleT, mean_inversion, txt_feat):

        # videos reshape
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc

        ts = (sampleT) / self.video_length
        ts = ts - ts[0]

        # downsample res for vae TODO: experiment with downsampling the resolution much more/ No downsampling
        vid_rs = nn.functional.interpolate(video_sample, size=(256//2, 192//2),  mode="bicubic", align_corners=False) # T*B x C x H//2 x W//2 
        vid_rs_tf = vid_rs.view(T, bs, ch, vid_rs.shape[2], vid_rs.shape[3])
        vid_rs_bf = vid_rs_tf.permute(1,0,2,3,4).contiguous() # B x T x C x H//2 x W//2


        # repeat text features 
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(T, 1, 1) # T x B x D
        txt_feat = txt_feat_tf.contiguous().view(T*bs, -1)  # T*B x D
        txt_feat.requires_grad = False

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs_bf, ts)
        video_style = zs
        video_dynamics = zd
                               

        # frame rep (video_style, video_content, dynamics)
        frame_rep = (txt_feat, video_dynamics, None) # T*B x D1+D2


        src_inversion_tf = mean_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, src_inversion_tf.shape[-2], src_inversion_tf.shape[-1])
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, *frame_rep)

        ret = self.stylegan_G(w_latents)
        ret = ret.reshape(T, bs, ret.shape[1], ret.shape[2], ret.shape[3]).permute(1, 0, 2, 3, 4)
        return ret

    def training_step(self, batch, batch_idx):
        input_desc = batch['raw_desc'] # B

        # videos reshape
        vid_bf = batch['real_img'] # B x T x C x H x W 
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc

        sampleT = batch['sampleT']  # B x T 
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        ts = (sampleT) / self.video_length
        ts = ts - ts[0]

        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        # inversions = inversions_tf.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D
        # inversions.requires_grad = False

        inverted_vid_bf = batch['inverted_img'] # B x T x ch x H x W -- range [0, 1]
        inverted_vid_tf = inverted_vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        inverted_vid = inverted_vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W 
        inverted_vid_norm = inverted_vid * 2 - 1 # range [-1, 1] to pass to the generator and disc
        

        # downsample res for vae TODO: experiment with downsampling the resolution much more/ No downsampling
        vid_rs = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True) # T*B x C x H//2 x W//2 
        vid_rs_tf = vid_rs.view(T, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs_bf = vid_rs_tf.permute(1,0,2,3,4).contiguous() # B x T x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D

        # repeat text features 
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(T, 1, 1) # T x B x D
        txt_feat = txt_feat_tf.contiguous().view(T*bs, -1)  # T*B x D
        txt_feat.requires_grad = False

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs_bf, ts)
        video_style = zs
        video_dynamics = zd
                               
        total_loss = 0
        vgg_loss = 0 
        beta_vae_loss = 0
        G_loss = 0

        # roll batch-wise
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=2) # T*B x D2
        
        # frame rep (video_style, video_content, dynamics)
        frame_rep = (txt_feat, video_dynamics, None) # T*B x D1+D2
        frame_rep_txt_mismatched = (txt_feat_mismatch, video_dynamics, None) # T*B x D1+D2

        # predict latents delta
        mean_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        src_inversion_tf = mean_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, n_channels, dim)
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, *frame_rep)
        w_latents_txt_mismatched = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, *frame_rep_txt_mismatched)

        reconstruction = self.stylegan_G(w_latents) # T*B x 3 x H x W
        imgs_txt_mismatched = self.stylegan_G(w_latents_txt_mismatched) # T*B x 3 x H x W

        reconstruction_inp_res = nn.functional.interpolate(reconstruction, size=(256, 192), mode="bicubic", align_corners=False)
        imgs_txt_mismatched_inp_res = nn.functional.interpolate(imgs_txt_mismatched, size=(256, 192), mode="bicubic", align_corners=False)



        # calculate losses
        # TODO: URGENT: experiment working on the full resolution
        reconstruction_loss = self.rec_loss(reconstruction_inp_res, inverted_vid_norm)


        # latent_loss = self.l2_latent_loss(src_inversion, w_latents) 
        # latent_loss += 2 * self.l2_latent_loss(src_inversion, w_latents_txt_mismatched)
        latent_loss = torch.maximum(self.l2_latent_loss(src_inversion, w_latents) - self.l2_latent_eps / 2, torch.zeros(1).to(src_inversion.device)[0]) 
        latent_loss += torch.maximum(self.l2_latent_loss(src_inversion, w_latents_txt_mismatched) - self.l2_latent_eps, torch.zeros(1).to(src_inversion.device)[0])
        vgg_loss = self.criterionVGG(reconstruction_inp_res.contiguous() / 2 + 0.5, inverted_vid.contiguous().detach()).mean()
        
        
        # video based losses
        txt_feat_bf = txt_feat_tf.permute(1, 0, 2).contiguous() # B x T x D

        
        txt_feat_mismatch_tf = txt_feat_mismatch.contiguous().reshape(T, bs, txt_feat_mismatch.shape[1])
        txt_feat_mismatch_bf = txt_feat_mismatch_tf.permute(1, 0, 2).contiguous()  # B x T x D

        vid_norm_bf = vid_bf * 2 - 1

        reconstruction_tf = reconstruction.contiguous().reshape(T, bs, reconstruction.shape[1], reconstruction.shape[2], reconstruction.shape[3])
        reconstruction_bf = reconstruction_tf.permute(1, 0, 2, 3, 4).contiguous()  # B x T x C x H x W 

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

        # lambdas
        self.log("lambda/consistency_loss", self.consistency_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/vgg_loss", self.lambda_vgg(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/rl_loss", self.rec_loss_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/clip_loss", self.clip_loss_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/l2_latent_loss", self.l2_latent_lambda(self.global_step) , prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # losses
        self.log("train/consistency_loss", consistency_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train/vgg_loss", vgg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train/rl_loss", reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/clip_loss", directional_clip_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/l2_latent_loss", latent_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        total_loss = self.rec_loss_lambda(self.global_step) * reconstruction_loss \
                    + self.clip_loss_lambda(self.global_step) * directional_clip_loss \
                    + self.l2_latent_lambda(self.global_step) * latent_loss \
                    + self.lambda_vgg(self.global_step) * vgg_loss \
                    + self.consistency_lambda(self.global_step) * consistency_loss
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return total_loss


    def validation_step(self, batch, batch_idx):
        pass


    def log_images(self, batch, split):
        """
        return a dictionary of tensors in the range [-1, 1]
        """
        ret = dict()

        input_desc = batch['raw_desc'] # B

        # videos reshape
        vid_bf = batch['real_img'] # B x T x C x H x W 
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        video_sample = vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc

        sampleT = batch['sampleT']  # B x T 
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1])
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        ts = (sampleT) / self.video_length
        ts = ts - ts[0]
        
        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        # inversions = inversions_tf.contiguous().reshape(T * bs, n_channels, dim) # T * B x n_layers x D
        # inversions.requires_grad = False

        inverted_vid_bf = batch['inverted_img'] # B x T x ch x H x W -- range [0, 1]
        inverted_vid_tf = inverted_vid_bf.permute(1,0,2,3,4) # T x B x C x H x W 
        inverted_vid = inverted_vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W 
        inverted_vid_norm = inverted_vid * 2 - 1 # range [-1, 1] to pass to the generator and disc

        # downsample res for vae
        vid_rs_full = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
        vid_rs = vid_rs_full.view(T, bs, ch, int(height*0.5),int(width*0.5) )
        vid_rs = vid_rs.permute(1,0,2,3,4) #  B x T x C x H//2 x W//2

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D

        # repeat text features 
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(T, 1, 1) # T x B x D
        txt_feat = txt_feat_tf.contiguous().view(T*bs, -1)  # T*B x D
        txt_feat.requires_grad = False

        # vae encode frames
        zs, zd, mu_logvar_s, mu_logvar_d = self.bVAE_enc(vid_rs, ts)
        video_style = zs
        video_dynamics = zd

        
        # roll batch-wise
        mismatch_txt_feat, _ = self.preprocess_text_feat(txt_feat, mx_roll=2)
        # mismatched_video_style, _ = self.preprocess_text_feat(video_style, mx_roll=2)


        # frame rep (video_style, video_content, dynamics)
        # frame_rep = (txt_feat, video_style, video_dynamics) # T*B x D1+D2
        # frame_rep_txt_mismatched = (mismatch_txt_feat, video_style, video_dynamics) # T*B x D1+D2
                # frame rep (video_style, video_content, dynamics)
        frame_rep = (txt_feat, video_dynamics, None) # T*B x D1+D2
        frame_rep_txt_mismatched = (mismatch_txt_feat, video_dynamics, None) # T*B x D1+D2

        # predict latents delta
        src_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        src_inversion_tf = src_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, n_channels, dim)
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, *frame_rep)
        w_latents_txt_mismatched = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, *frame_rep_txt_mismatched)

        ret['real_image'] = video_sample_norm 
        ret['inverted_image'] = inverted_vid_norm 
        ret['real_image_caption'] = '\n'.join([f"{batch['video_name'][i]}: {el}" for i, el in enumerate(batch['raw_desc'])])
        ret['x_recon_gan']  = self.stylegan_G(w_latents)
        ret['x_mismatch_text']  = self.stylegan_G(w_latents_txt_mismatched)

        ret['x_recon_gan'] = nn.functional.interpolate(ret['x_recon_gan'], size=self.frame_log_size, mode="nearest")
        ret['x_mismatch_text'] = nn.functional.interpolate(ret['x_mismatch_text'], size=self.frame_log_size, mode="nearest")

        if split == 'val' and self.global_step > 0:
            C, H, W = ret['real_image'].shape[1:]
             # TODO: create gif instead of frames 
            self.log_gif(vid_tf, range=(0, 1), name='val/real_gif')
            self.log_gif(inverted_vid_tf, range=(0, 1), name='val/inverted_gif')           
            self.log_gif(ret['x_recon_gan'].reshape(T, bs, C, H , W).contiguous(), range=(-1, 1), name='val/recon_gif')
            self.log_gif(ret['x_mismatch_text'].reshape(T, bs, C, H , W).contiguous(), range=(-1, 1), name='val/mismatch_gif')
        return ret

    
    def log_gif(self, video, range, name):
        # Assuming that the current shape is T * B x C x H x W
        filename = f'tmp/{str(uuid.uuid4())}.gif'
        with imageio.get_writer(filename, mode='I') as writer:
           for b_frames in video:
                # b_frames B x C x H x W
                frame = torchvision.utils.make_grid(b_frames,
                                nrow=b_frames.shape[0],
                                normalize=True,
                                range=range).detach().cpu().numpy()
                frame = (np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8)
                writer.append_data(frame)

        wandb.log({name: wandb.Video(filename, fps=2, format="gif")})
        os.remove(filename)




    def configure_optimizers(self):
        lr = self.learning_rate
        vae_params = list(self.bVAE_enc.parameters())

        style_m_params = list(self.style_mapper.parameters())
        
        opt_vae = torch.optim.Adam(vae_params,
                                  lr=lr * 5, 
                                  betas=(0.9, 0.999))

        opt_sm = torch.optim.Adam(style_m_params, lr=lr, betas=(0.9, 0.99))

        #opt_m = torch.optim.Adam(m_params,
        #                          lr=lr / 100, 
        #                          betas=(0.5, 0.999))

        #opt_ae = HybridOptim([opt_vae, opt_m, opt_sm])
        opt_ae = HybridOptim([opt_vae, opt_sm])
        
        
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
