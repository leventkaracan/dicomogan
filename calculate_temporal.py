import numpy as np
import torch
from torchvision import transforms
from metric_scripts.metric_dataloader import MetricsDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

from asenpp.modules.config import cfg
from asenpp.modules.model import build_model



path_to_real_images = "/scratch/users/abond19/datasets/aligned_fashion_dataset/"
path_to_fake_images = "/scratch/users/abond19/datasets/inverted_fashion_dataset/"

model_to_use = "FashionAI"

model_weights_path = f"/scratch/users/abond19/VideoEditing/dicomogan/asenpp/{model_to_use}.pth.tar"
model_config_path = f"/scratch/users/abond19/VideoEditing/dicomogan/asenpp/config/{model_to_use}/{model_to_use}.yaml"

transform = transforms.Compose(
			[
				transforms.Resize(224),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]
				)
			]
		)

dataset = MetricsDataset(path_to_real_images, path_to_fake_images, transform=transform)

BATCH_SIZE = 1

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=5)

cfg.merge_from_file(model_config_path)
cfg.freeze()

number_of_attributes = len(cfg.DATA.ATTRIBUTES.NAME)

model = build_model(cfg)
model_weights = torch.load(model_weights_path)
model.load_state_dict(model_weights)
model.cuda()

real_tl_similarities = []
fake_tl_similarities = []

real_gl_similarities = []
fake_gl_similarities = []

output_file = open(f"temporal_metric_outputs_{model_to_use}.txt", "w")

with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader, total=len(dataset) // BATCH_SIZE)):
        real_image = batch['real_image'].float().cuda().squeeze() # T x C x H x W
        fake_image = batch['fake_image'].float().cuda().squeeze() # T x C x H x W
        
        if real_image.shape[0] != 15 or fake_image.shape[0] != 15:
            continue
        
        real_attribute_embeddings = []
        fake_attribute_embeddings = []

        for i in range(number_of_attributes):
            attr_vector = torch.LongTensor([i]).repeat(15).cuda() # T
            real_encoded = model(real_image, attr_vector)[0] # T x D
            fake_encoded = model(fake_image, attr_vector)[0] # T x D

            real_attribute_embeddings.append(real_encoded)
            fake_attribute_embeddings.append(fake_encoded)
        
        real_attribute_embeddings = torch.stack(real_attribute_embeddings)
        fake_attribute_embeddings = torch.stack(fake_attribute_embeddings)

        total_tl_real_similarities = []
        total_tl_fake_similarities = []

        # For TL-ID
        for i in range(number_of_attributes):
            total_real_similarity = 0
            total_fake_similarity = 0
            for j in range(1, real_attribute_embeddings.shape[1]):
                real_similarity = cosine_similarity(real_attribute_embeddings[i, j-1, :], real_attribute_embeddings[i, j, :], dim=-1).cpu().item()
                fake_similarity = cosine_similarity(fake_attribute_embeddings[i, j-1, :], fake_attribute_embeddings[i, j, :], dim=-1).cpu().item()
                total_real_similarity += real_similarity
                total_fake_similarity += fake_similarity
            
            total_tl_real_similarities.append(total_real_similarity / 14)
            total_tl_fake_similarities.append(total_fake_similarity / 14)

        real_tl_similarities.append(np.array(total_tl_real_similarities))
        fake_tl_similarities.append(np.array(total_tl_fake_similarities))

        total_gl_real_similarities = []
        total_gl_fake_similarities = []

        #For TG-ID
        for i in range(number_of_attributes):
            total_real_similarity = 0
            total_fake_similarity = 0
            counter = 0
            for j in range(1, real_attribute_embeddings.shape[1]):
                for k in range(j, real_attribute_embeddings.shape[1]):
                    if j == k:
                        continue
                    real_similarity = cosine_similarity(real_attribute_embeddings[i, j, :], real_attribute_embeddings[i, k, :], dim=-1).cpu().item()
                    fake_similarity = cosine_similarity(fake_attribute_embeddings[i, j, :], fake_attribute_embeddings[i, k, :], dim=-1).cpu().item()
                    total_real_similarity += real_similarity
                    total_fake_similarity += fake_similarity
                    counter += 1
            total_gl_fake_similarities.append(total_fake_similarity / counter)
            total_gl_real_similarities.append(total_real_similarity / counter)
        
        real_gl_similarities.append(np.array(total_gl_real_similarities))
        fake_gl_similarities.append(np.array(total_gl_fake_similarities))


real_tl_similarities = np.array(real_tl_similarities)
fake_tl_similarities = np.array(fake_tl_similarities)

real_tl_similarities = real_tl_similarities.mean(axis=0)
fake_tl_similarities = fake_tl_similarities.mean(axis=0)

real_gl_similarities = np.array(real_gl_similarities)
fake_gl_similarities = np.array(fake_gl_similarities)

real_gl_similarities = real_gl_similarities.mean(axis=0)
fake_gl_similarities = fake_gl_similarities.mean(axis=0)

output_file.write("TL-ID:\n----------------------------\n\n")
for i in range(number_of_attributes):
    output_file.write(f"Average real cosine similarity for {cfg.DATA.ATTRIBUTES.NAME[i]}: {np.array(real_tl_similarities[i]).mean()}\n")
    output_file.write(f"Average fake cosine similarity for {cfg.DATA.ATTRIBUTES.NAME[i]}: {np.array(fake_tl_similarities[i]).mean()}\n\n")

output_file.write("\n\nTG-ID:\n----------------------------\n\n")
for i in range(number_of_attributes):
    output_file.write(f"Average real cosine similarity for {cfg.DATA.ATTRIBUTES.NAME[i]}: {np.array(real_gl_similarities[i]).mean()}\n")
    output_file.write(f"Average fake cosine similarity for {cfg.DATA.ATTRIBUTES.NAME[i]}: {np.array(fake_gl_similarities[i]).mean()}\n\n")

output_file.close()