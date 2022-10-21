import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm
from fvd.fvd_utils import frechet_distance, get_fvd_logits

from metric_scripts.metric_dataloader import MetricsDataset

from fvd.fvd_utils import *

# Assuming that all real images and all fake images are stored in folders
path_to_real_images = "/scratch/users/abond19/datasets/aligned_fashion_dataset/"
path_to_fake_images = "/kuacc/users/abond19/datasets/model_generated_dataset/_w-vidode-irregualar-sampling2022-10-20T11-15-48/"
#path_to_fake_images = "/scratch/users/abond19/datasets/inverted_fashion_dataset/"

dataset = MetricsDataset(path_to_real_images, path_to_fake_images)

BATCH_SIZE = 1
TARGET_RESOLUTION = (224, 224)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

real_activations = []
fake_activations = []

i3d = load_fvd_model(torch.device("cuda"))

for idx, pair in enumerate(tqdm(dataloader, total=len(dataset) // BATCH_SIZE)):
    real = pair['real_image'].numpy()
    fake = pair['fake_image'].numpy()

    if real.shape[1] < 15 or fake.shape[1] < 15:
        continue

    real_activation = get_fvd_logits(real.transpose(0, 1, 3, 4, 2), i3d, torch.device("cuda")).cpu()
    fake_activation = get_fvd_logits(fake.transpose(0, 1, 3, 4, 2), i3d, torch.device("cuda")).cpu()

    real_activations.append(real_activation)
    fake_activations.append(fake_activation)



real_activations = torch.cat(real_activations, dim=0)
fake_activations = torch.cat(fake_activations, dim=0)

print(frechet_distance(real_activations, fake_activations))