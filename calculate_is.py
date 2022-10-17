from torchmetrics.image.inception import InceptionScore
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader

from tqdm import tqdm

from metric_scripts.metric_dataloader import MetricsDataset

real_inception = InceptionScore().cuda()
fake_inception = InceptionScore().cuda()

# Assuming that all real images and all fake images are stored in folders
path_to_real_images = "/scratch/users/abond19/datasets/aligned_fashion_dataset/"
path_to_fake_images = "/scratch/users/abond19/datasets/inverted_fashion_dataset/"

dataset = MetricsDataset(path_to_real_images, path_to_fake_images)

BATCH_SIZE = 1

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

for pair in tqdm(dataloader, total=len(dataset) // BATCH_SIZE):
    real = pair['real_image'].cuda()
    fake = pair['fake_image'].cuda()
    
    if real.shape[1] < 15 or fake.shape[1] < 15:
        continue

    B, T, C, H, W = real.shape
    real = real.reshape(B * T, C, H, W)
    fake = fake.reshape(B * T, C, H, W)
    
    real_inception.update(real)
    fake_inception.update(fake)

print(real_inception.compute())
print(fake_inception.compute())