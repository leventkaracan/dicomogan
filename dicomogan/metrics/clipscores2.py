import clip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import random
import imageio
import clip


device = "cuda"
#descriptions = open("/data/levent/target-descriptions_3dshapes.txt").readlines()
descriptions = open( "/data/levent/fashion_datasetv3/fashion_dataset/metadata/target_descriptions_fashion.txt").readlines()
model, preprocess = clip.load("ViT-B/32", device=device)
input_dir = "/data/levent/results_fashion_textvaeode2im_ganvgg/input/"
#input_dir = "/data/levent/results_fashionNew/input/"

#dir = "/home/levent/sisgan/examples/target/"
#dir = "/data/levent/manigan/output/3dshapes/netG_epoch_85/target/"
#dir = "/data/levent/results_3dshapes_GANUnsup/target/"
#dir = "/data/levent/results_fashionNew/target/"
#dir = "/data/levent/results_3dshapes_sisgan/target/"
dir = "/data/levent/results_fashion_textvaeode2im_adain/target/"

l1_loss = nn.L1Loss()
cos_sim = nn.CosineSimilarity()
scores = []
filet = open("fashion_ganvgg_clip_mp.txt", "w")
for i in range(600):
  for j in range(12):
    input_image = preprocess(Image.open(input_dir + f"{i}_{j}.png")).unsqueeze(0).to(device)
    image = preprocess(Image.open(dir + f"{i}_{j}.png")).unsqueeze(0).to(device)
    image_norm = (image - image.min())/(image.max() - image.min())
    input_image_norm = (input_image - input_image.min())/(input_image.max() - input_image.min())

    text = clip.tokenize(descriptions[i]).to(device)
    
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    #print(text_features-image_features)
    #print(image_features)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T)
    #print(similarity)
    
    diff = l1_loss(input_image_norm,image_norm)
    #print(diff)
    scores.append((1-diff.item())*similarity[0][0].item())
    filet.write(str(np.mean(np.array(scores))) + "\n")
    print(np.mean(np.array(scores)))
    filet.flush()
  print(i)
print(np.mean(np.array(scores)))
