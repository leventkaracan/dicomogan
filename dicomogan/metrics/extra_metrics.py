import clip
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch.nn as nn
import random
import torch

def clip_forward(imgs, text):
    image_features = self.encode_image(image)
    text_features = text

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def clip_score(clip_loss, imgs, text, fake_img):
    device = "cuda"
    #model, preprocess = clip.load("ViT-B/32", device=device)

    new_preprocess = transforms.Compose(clip_loss.clip_preprocess.transforms[1:])

    scores = []

    num_imgs = imgs.shape[0]

    for i in range(num_imgs):
        #current_text = clip.tokenize(text[i]).to(device)
        current_text = text[i]
        if fake_img:
            current_img = clip_loss.clip_preprocess(imgs[i])
        else:
            current_img = new_preprocess(imgs[i])
        
        image_features = clip_loss.model.encode_image(current_img.unsqueeze(0))
        text_features = text[i].unsqueeze(0)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = clip_loss.model.logit_scale.exp()
        #print(logit_scale.dtype, image_features.dtype, text_features.dtype)
        logits_per_image = logit_scale * image_features.to(dtype=torch.float32) @ text_features.t()
        logits_per_text = logits_per_image.t()

        scores.append(logits_per_image.item()/100)
    
    return np.mean(np.array(scores))

#def clip_score_2(imgs, text):

