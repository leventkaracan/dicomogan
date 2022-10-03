import clip
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch.nn as nn
import random

def clip_score(imgs, text):
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    scores = []

    B, n_frames, C, H, W = imgs.shape
    imgs = imgs.permute(1, 0, 2, 3, 4)
    for i in range(n_frames):
        current_text = clip.tokenize(text[i]).to(device)
        logits_per_image, logits_per_text = model(imgs[i], current_text)
        scores.append(logits_per_image.item()/100)
    
    return np.mean(np.array(scores))

#def clip_score_2(imgs, text):

