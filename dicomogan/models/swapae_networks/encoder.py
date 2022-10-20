import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from dicomogan.models.swapae_networks import BaseNetwork
from dicomogan.models.swapae_networks.stylegan2_layers import ResBlock, ConvLayer, ToRGB, EqualLinear, Blur, Upsample, make_kernel
from dicomogan.models.swapae_networks.stylegan2_op import upfirdn2d


def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

class ToSpatialCode(torch.nn.Module):
    def __init__(self, inch, outch, scale):
        super().__init__()
        hiddench = inch // 2
        self.conv1 = ConvLayer(inch, hiddench, 1, activate=True, bias=True)
        self.conv2 = ConvLayer(hiddench, outch, 1, activate=False, bias=True)
        self.scale = scale
        self.upsample = Upsample([1, 3, 3, 1], 2)
        self.blur = Blur([1, 3, 3, 1], pad=(2, 1))
        self.register_buffer('kernel', make_kernel([1, 3, 3, 1]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(int(np.log2(self.scale))):
            x = self.upsample(x)
        return x


class StyleGAN2ResnetEncoder(BaseNetwork):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
    #     parser.add_argument("--netE_num_downsampling_sp", default=4, type=int)
    #     parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
    #     parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
    #     return parser

    def __init__(self, netE_num_downsampling_sp, netE_num_downsampling_gl, spatial_code_ch, global_code_ch, netE_nc_steepness=2.0, netE_scale_capacity=1.0):
        super().__init__()
        self.netE_num_downsampling_sp = netE_num_downsampling_sp
        self.netE_num_downsampling_gl = netE_num_downsampling_gl
        self.spatial_code_ch = spatial_code_ch
        self.global_code_ch = global_code_ch
        self.netE_nc_steepness = netE_nc_steepness
        self.netE_scale_capacity = netE_scale_capacity
        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
            )
        )

        self.DownToGlobalCode = nn.Sequential()
        for i in range(netE_num_downsampling_gl):
            idx_from_beginning = netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )

        nchannels = self.nc(netE_num_downsampling_sp +
                            netE_num_downsampling_gl)
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, global_code_ch)
            )
        )

    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        nc = min(self.global_code_ch, int(round(nc)))
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.DownToGlobalCode[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.DownToGlobalCode(midpoint)
        x = x.mean(dim=(2, 3)) # global avg
        gl = self.ToGlobalCode(x)
        sp = normalize(sp)
        gl = normalize(gl)
        if extract_features:
            return sp, gl, feature
        else:
            return sp, gl
