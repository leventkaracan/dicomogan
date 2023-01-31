
import sys 
sys.path.append('StyleGAN_Human')
sys.path.append('.')

import torch.nn as nn
import torch
import numpy as np
from StyleGAN_Human import dnnlib, legacy
# from StyleGAN_Human.torch_utils.models import Generator as FashionGenerator
from StyleGAN_Human.stylegan2.model import Generator as FashionGenerator
from mocogan_stylegan2 import Generator as MoCoGenerator
import legacy2 as legacy_e4e
import misc2 as misc_e4e
import os
from models.stylegan2.model import Generator
from models.stylegan2.model_e4e import Generator as Generator_e4e
from torch_utils import misc

class StyleGAN2Generator(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        with dnnlib.util.open_url(pkl_file) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].eval()
            self.G = self.G.float()
    
    def forward(self, w):
        return self.G.synthesis(w, noise_mode='const', force_fp32=True)

    def synthesis(self, device):
        label = torch.zeros([1, self.G.c_dim]).to(device)
        z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(device)
        w = self.G.mapping(z, label)
        print(w.requires_grad)
        return self.forward(w)

class StyleGAN2Face_FT(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator_e4e(1024, 512, 8, channel_multiplier=2)

        with dnnlib.util.open_url(pkl_file) as f:
            ckpt = legacy_e4e.load_network_pkl(f, force_fp16=True)

        for name, module in [('G_ema', self.G)]:
            misc_e4e.copy_params_and_buffers(ckpt[name], module, require_all=False)

        self.G = self.G.eval().float()

    def forward(self, w):
        # self.G([w], input_is_latent=True, randomize_noise=False, return_latents=False)
        imgs, _ =  self.G([w], input_is_latent=True, return_latents=True, randomize_noise=False)
        return imgs 


class StyleGAN2Sky(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = MoCoGenerator(128, 512, 8)
        pkl_obj = torch.load(pkl_file, map_location='cpu')
        if 'g' in pkl_obj:
            pkl_obj = pkl_obj['g']
        elif 'G_ema' in pkl_obj:
            pkl_obj = pkl_obj['G_ema']
        elif 'g_ema' in pkl_obj:
            pkl_obj = pkl_obj['g_ema']
        else:
            raise "cannot find appropriate key"
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w, return_latents=False, input_is_latent=True):
        out =  self.G([w], input_is_latent=input_is_latent, return_latents=True, randomize_noise=False)
        if return_latents:
            return out
        else: 
            return out[0]



class StyleGAN2TaiChi(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator(256, 512, 2, channel_multiplier=1)
        pkl_obj = torch.load(pkl_file, map_location='cpu')
        if 'g' in pkl_obj:
            pkl_obj = pkl_obj['g']
        elif 'G_ema' in pkl_obj:
            pkl_obj = pkl_obj['G_ema']
        elif 'g_ema' in pkl_obj:
            pkl_obj = pkl_obj['g_ema']
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]

class StyleGAN2Face(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator(1024, 512, 8)
        pkl_obj = torch.load(pkl_file, map_location='cpu')
        if 'g' in pkl_obj:
            pkl_obj = pkl_obj['g']
        elif 'G_ema' in pkl_obj:
            pkl_obj = pkl_obj['G_ema']
        elif 'g_ema' in pkl_obj:
            pkl_obj = pkl_obj['g_ema']
        else:
            raise "cannot find appropriate key"
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]
    
class StyleGAN2Fashion(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = FashionGenerator(1024, 512, 8)
        pkl_obj = torch.load(pkl_file, map_location='cpu')
        if 'g_ema' in pkl_obj:
            pkl_obj = pkl_obj['g_ema']
        else:
            raise "cannot find appropriate key"
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]


class StyleGAN2Cat(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator(512, 512, 8)
        self.G.load_state_dict(torch.load(pkl_file, map_location='cpu')['g_ema'], strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]
    
    def synthesis(self, device):
        z = torch.from_numpy(np.random.randn(1, self.G.style_dim)).to(device).float()
        img = self.G([z],  randomize_noise=True)[0]
        return img