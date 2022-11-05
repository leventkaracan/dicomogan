
import sys 
sys.path.append('StyleGAN_Human')
sys.path.append('.')

import torch.nn as nn
import torch
import numpy as np
from StyleGAN_Human import dnnlib, legacy
import os
from models.stylegan2.model import Generator

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


class StyleGAN2Face(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator(1024, 512, 8)
        self.G.load_state_dict(torch.load(pkl_file, map_location='cpu')['g'], strict=True)
        self.G.float().eval()
        for parameter in self.G.parameters():
            parameter.requires_grad = False
    
    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]