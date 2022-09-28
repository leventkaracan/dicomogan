import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import clip
from PIL import Image

#TODO: Figure out why this code only works when batch size is 1

# Taken from Rosinality StyleGAN2
# https://github.com/rosinality/stylegan2-pytorch/blob/3dee637b8937bf3830991c066ed8d9cc58afd661/model.py
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# Taken from HairCLIP 
# https://github.com/wty-ustc/HairCLIP/blob/main/mapper/latent_mappers.py
class ModulationModule(nn.Module):
    def __init__(self, layernum, attr_vec_dim):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.attr_vec_dim = attr_vec_dim
        self.gamma_function = nn.Sequential(nn.Linear(attr_vec_dim, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function = nn.Sequential(nn.Linear(attr_vec_dim, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embedding, cut_flag):
        # print(f"x before fc: {x.shape}")
        # print(f"Embedding: {embedding.shape}")
        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag:
            return x
        # print(embedding.shape)
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())

        # print(f"x: {x.shape}")
        # print(f"gamma: {gamma.shape}")
        # print(f"beta: {beta.shape}")
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out
    
class AttributeMapperModule(nn.Module):
    def __init__(self, attr_vec_dim, layernum):
        super(AttributeMapperModule, self).__init__()
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum, attr_vec_dim) for i in range(5)])

    def forward(self, x, embedding, cut_flag=False):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
        	x = modulation_module(x, embedding, cut_flag)        
        return x
    
class AttributeMapper(nn.Module):
    def __init__(self,
                use_coarse_mapper,
                use_medium_mapper,
                use_fine_mapper,
                coarse_cut_flag,
                medium_cut_flag,
                fine_cut_flag,
                attr_vec_dim):
        super(AttributeMapper, self).__init__()
        self.use_coarse_mapper = use_coarse_mapper
        self.use_medium_mapper = use_medium_mapper
        self.use_fine_mapper = use_fine_mapper
        self.coarse_cut_flag = coarse_cut_flag
        self.medium_cut_flag = medium_cut_flag
        self.fine_cut_flag = fine_cut_flag

        self.attr_vec_dim = attr_vec_dim
        
        # Original HairCLIP includes CLIP here
        # For the proposed attribute manipulation model, we don't need any CLIP model, so we can ignore it
        # For the proposed text-guided manipulation model, we would need CLIP in other places as well, so 
        # we don't put it here, and instead just pass the CLIP embeddings into the model
        
        # Currently set to use 3 mappers (one for coarse, medium, and fine)
        # TODO: Compare these 3 with using one for each layer
        # TODO: Remember to pass the dimension of the attribute vector in opts
        if self.use_coarse_mapper: 
            self.coarse_mapping = AttributeMapperModule(self.attr_vec_dim, 4)
        if self.use_medium_mapper:
            self.medium_mapping = AttributeMapperModule(self.attr_vec_dim, 4)
        if self.use_fine_mapper:
            self.fine_mapping = AttributeMapperModule(self.attr_vec_dim, 10)
            
        # TODO Figure out use of this
        # self.face_pool = nn.AdaptiveAvgPool2d((224, 224))
        # self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        
            
    # TODO: Figure out why preprocess is a parameter here if we never use it?
    # def gen_image_embedding(self, img_tensor, clip_model, preprocess):
    #     masked_generated = self.face_pool(img_tensor)
    #     masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)
    #     return clip_model.encode_image(masked_generated_renormed)
    
    # TODO: Figure out the input shape of x (unclear currently)
    def forward(self, x, attribute_vector):
        if attribute_vector.shape[1] != 1:
            attribute_vector = attribute_vector.unsqueeze(1).repeat(1, 18, 1)


        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]
        
        # All this assumes that we are using the medium group for the attribute. Change as necessary
        #A, B, C = x_medium.shape
        # Do we need to reshape attribute_vector at all, or is a 1-dimensional vector good enough
        #attribute_repeated = attribute_vector.unsqueeze(0).unsqueeze(0).repeat(A, B, 1)
        
        # TODO: Remember to add coarse, medium, and fine cut flags to the opts
        if self.use_coarse_mapper:
            x_coarse = self.coarse_mapping(x_coarse, attribute_vector[:, :4, :], cut_flag=self.coarse_cut_flag)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if self.use_medium_mapper:
            x_medium = self.medium_mapping(x_medium, attribute_vector[:, 4:8, :], cut_flag=self.medium_cut_flag)
        else:
            x_medium = torch.zeros_like(x_medium)
        if self.use_fine_mapper:
            x_fine = self.fine_mapping(x_fine, attribute_vector[:, 8:, :], cut_flag=self.fine_cut_flag)
        else:
            x_fine = torch.zeros_like(x_fine)
            
        output = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return output