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
model, preprocess = clip.load("ViT-B/32", device=device)
descriptions = open("/data/levent/target-descriptions_3dshapes.txt").readlines()
input_dir = "/data/levent/results_3dshapes_GANLoss/input/"
#dir = "/home/levent/sisgan/examples/target/"
dir = "/data/levent/results_3dshapes_adain/target/"
scores = []
filet = open("adain_clip.txt", "w")
for i in range(4799):
  for j in range(15):
    image = preprocess(Image.open(dir + f"{i}_{j}.png")).unsqueeze(0).to(device)
    text = clip.tokenize(descriptions[i]).to(device)
    logits_per_image, logits_per_text = model(image, text)
    scores.append(logits_per_image.item()/100)
    filet.write(str(np.mean(np.array(scores))) + "\n")
    print(np.mean(np.array(scores)))
    filet.flush()
  print(i)
print(np.mean(np.array(scores)))
