from skimage import io, transform
import os
import numpy as np
from PIL import Image
from nltk.tokenize import RegexpTokenizer
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pickle
import torchvision.transforms.functional as func
import random
import glob

IMG_EXTENSIONS = ['.png', '.PNG']
TXT_EXTENSIONS = ['.txt']

def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)

class VideoDataFashion(data.Dataset):
    def __init__(self, video_list, img_root, n_sampled_frames, 
                    batch_size, img_transform=None, 
                    inversion_root=None,
                    crop=(512, 384), size=(256, 192)):
                    
        super(VideoDataFashion, self).__init__()
        self.img_transform = img_transform
        if self.img_transform is None:
            self.img_transform=transforms.Compose([
                          transforms.CenterCrop(tuple(crop)),
                          transforms.Resize(tuple(size)),
                          transforms.ToTensor()
                      ])
        
        self.img_root = img_root
        self.video_list = open(video_list).readlines()
        self.inversion_root = inversion_root
        self.n_sampled_frames = n_sampled_frames   
        self.batch_size = batch_size
        self.data_paths, self.inversion_paths, self.desc_paths, self.frame_numbers = self._load_dataset()        

    def _load_dataset(self):
        data_paths, inversion_paths, desc_paths, frame_numbers = [], [], [], []
        intersection = None
        for idx, vid_path in enumerate(self.video_list):
            paths, i_paths, d_paths, f_nums = [], [], [], []
            fname = vid_path[:-1]
            for f in sorted(os.listdir(os.path.join(self.img_root, fname))):
                if is_image_file(f):
                    imname = f[:-4]
                    paths.append(os.path.join(self.img_root, fname, f))
                    f_nums.append(int(imname))

                    if self.inversion_root is not None:
                        i_paths.append(os.path.join(os.path.join(self.inversion_root, fname, imname + ".pt")))
                elif is_text_file(f):
                    d_paths.append(os.path.join(self.img_root, fname, f))
            
            data_paths.append(sorted(paths))
            inversion_paths.append(sorted(i_paths))
            desc_paths.append(sorted(d_paths))
            intersection = set(sorted(f_nums)) if intersection is None else intersection.intersection(set(sorted(f_nums)))
            if (idx+1) % self.batch_size == 0:
                frame_numbers.append(sorted(list(intersection)))
                intersection = None

        if intersection is not None:
            frame_numbers.append(sorted(list(intersection)))
        return data_paths, inversion_paths, desc_paths, frame_numbers

    def __getitem__(self, index):
        # load text 
        rnd_txt = np.random.randint(len(self.desc_paths[index]))
        raw_desc = open(self.desc_paths[index][rnd_txt]).readlines()[0]

        # sample frames
        bin = index // self.batch_size
        local_state = np.random.RandomState(bin + 1)
        sampleT = local_state.choice(self.frame_numbers[bin][1:], self.n_sampled_frames-1, replace=False)
        sampleT = np.append(sampleT, self.frame_numbers[bin][0])
        sampleT = np.sort(sampleT)

        I, W = None, None
        for i in sampleT:
            Ii = self.get_image(self.data_paths[index][i])
            Ii = self.img_transform(Ii)
            Ii = torch.unsqueeze(Ii, 0)
            I = Ii if I is None else torch.cat([I, Ii], dim=0)

            # Getting the inversion
            if self.inversion_root is not None:
                assert self.data_paths[index][i][len(self.img_root):].split('.')[0] == self.data_paths[index][i][len(self.img_root):].split('.')[0], "inverstion does not matched image"
                w_path = self.inversion_paths[index][i]
                w_vec = self.get_inversion(w_path)
                W = w_vec if W is None else torch.cat([W, w_vec], dim=0)
        
        return_list = {'img': I, 'raw_desc': raw_desc, "sampleT": sampleT, "index": index}
        if W is not None:
            return_list['inversion'] = W

        return return_list


    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def get_inversion(self, inversion_path):
        w_vector = torch.load(inversion_path)
        assert (w_vector.shape == (1, 18, 512)), "Inverted vector has incorrect shape"
        return w_vector
    
    def __len__(self):
        return len(self.data_paths)
    
    def name(self):
        return 'VideoDataFashion'
    

# TODO: Add the modifications for the test dataset as well