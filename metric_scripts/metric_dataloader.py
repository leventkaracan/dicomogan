import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os

# Here, I assume that the real and fake images have been stored in .npy files, but we can easily change that if necessary
class MetricsDataset(Dataset):
    def __init__(self, path_to_real_images, path_to_fake_images):
        super().__init__()

        self.path_to_real_images = path_to_real_images
        self.path_to_fake_images = path_to_fake_images

        self.real_images, self.fake_images = self._load_dataset()

        self.transform = transforms.ToTensor()


    def _load_dataset(self):
        folder_names = os.listdir(self.path_to_real_images)

        real_images = []
        fake_images = []

        for folder in folder_names:
            if not os.path.exists(os.path.join(self.path_to_fake_images, folder)):
                continue
                
            current_real_images = []
            current_fake_images = []

            real_folder_path = os.path.join(self.path_to_real_images, folder)
            fake_folder_path = os.path.join(self.path_to_fake_images, folder)

            fake_files = os.listdir(fake_folder_path)

            for file in os.listdir(real_folder_path):
                if file[-4:] != ".png":
                    continue

                if file not in fake_files:
                    continue

                current_real_images.append(os.path.join(real_folder_path, file))
                current_fake_images.append(os.path.join(fake_folder_path, file))
            
            real_images.append(current_real_images)
            fake_images.append(current_fake_images)
        
        return real_images, fake_images

    def __getitem__(self, index):
        real_images = []
        fake_images = []

        for image_path in self.real_images[index]:
            img = (255 * self.transform(Image.open(image_path))).to(torch.uint8)
            real_images.append(img)

        for image_path in self.fake_images[index]:
            img = (255 * self.transform(Image.open(image_path))).to(torch.uint8)
            fake_images.append(img)
        
        real_image = torch.stack(real_images)
        fake_image = torch.stack(fake_images)

        ret_dict = dict()
        ret_dict["real_image"] = real_image[5:20]
        ret_dict["fake_image"] = fake_image[5:20]

        return ret_dict

    # Assuming that real_images and fake_images have the same shape, which should be true if the fake_images are just reconstructions of the real_images
    def __len__(self):
        return len(self.real_images)