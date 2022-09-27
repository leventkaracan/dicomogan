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

def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class VideoDataFashion(data.Dataset):
    def __init__(self, img_root, inversion_path, align_path, n_frames_total, batch_size, img_transform=None, crop=(512, 384), size=(256, 192)):
        super(VideoDataFashion, self).__init__()
        self.img_transform = img_transform
        if self.img_transform is None:
            self.img_transform=transforms.Compose([
                          transforms.CenterCrop(tuple(crop)),
                          transforms.Resize(tuple(size)),
                          transforms.ToTensor()
                      ])
        
        # TODO: Need to modify the metadata files since some files were removed
        self.img_root = img_root
        self.align_path = align_path
        self.video_list = open(img_root + "metadata/train_video.txt").readlines()
        #self.description_list = open(img_root + "metadata/train_video_descriptions.txt").readlines()
        self.multi_description_list = [open(img_root + "metadata/train_video_descriptions.txt").readlines(), open(img_root + "metadata/train_video_descriptions1.txt").readlines(), open(img_root + "metadata/train_video_descriptions2.txt").readlines(), open(img_root + "metadata/train_video_descriptions3.txt").readlines(), open(img_root + "metadata/train_video_descriptions4.txt").readlines(), open(img_root + "metadata/train_video_descriptions5.txt").readlines()]
        self.inversion_path = inversion_path
        # TODO: Need to create this file in the inversion folder metadata
        #self.inversion_list = open(inversion_path + "metadata/train_video.txt").readlines()
        self.data_paths, self.raw_descriptions, self.inversions = self._load_dataset()
        self.n_of_seqs = len(self.data_paths)                 # number of sequences to train       
        self.seq_len_max = max([len(V) for V in self.data_paths])        
        self.n_frames_total = n_frames_total      # current number of frames to train in a single iteration
        self.batch_size = batch_size

    # TODO: Currently assuming that every frame of every video has been inverted properly
    def _load_dataset(self):
        images = []
        descriptions_raw = []
        inversions = []
        count = 0
        descriptions_lists = self.multi_description_list
        for idx, vid_path in enumerate(self.video_list):
            desc_raw = []
            for e in range(6):
                description = descriptions_lists[e][idx]
                raw_desc = description[:-1]
                desc_raw.append(raw_desc)
            paths = []
            description = []
            description_raw = []
            inversion_paths = []
            #desc_len = []
            raw_desc = raw_desc[:-1]
            fname = vid_path[:-1]
            count = count + 1
            for f in sorted(os.listdir(self.img_root + fname))[:15]:
                if is_image_file(f):
                    imname = f[:-4]
                    # Replaced self.img_root with self.align_path so we always use the aligned version of each frame
                    paths.append(os.path.join(self.align_path, fname + "/" + f))
                    i_path = os.path.join(self.inversion_path, fname+"/"+ imname + ".pt")
                    inversion_paths.append(i_path)
                    description_raw.append(desc_raw)
            
            if len(paths) > 0:
                images.append(paths)
                descriptions_raw.append(description_raw)
                inversions.append(inversion_paths)
    
        return images, descriptions_raw, inversions

    def __getitem__(self, index):
        data_paths = self.data_paths[index]
        rnd_txt = np.random.randint(6)
        raw_desc = self.raw_descriptions[index][:][rnd_txt]
        num_frames = len(data_paths)

        inversion_paths = self.inversions[index]

        sampleT = np.arange(num_frames - 1) + 1
        local_state = np.random.RandomState(int(index // self.batch_size) + 1)
        np.random.seed(int(index // self.batch_size) + 1)
        sampleT = np.random.choice(sampleT, self.n_frames_total, replace=False)
        sampleT = np.sort(sampleT)

        I = None
        framelist = []
        inversion_list = []
        W = None
        for i in sampleT:
            # Getting the actual image
            I_path = data_paths[i]
            Ii = self.get_image(I_path)
            Ii = self.img_transform(Ii)
            Ii = torch.unsqueeze(Ii, 0)
            I = Ii if I is None else torch.cat([I, Ii], dim=0)
            framelist.append(data_paths[i])

            # Getting the inversion
            w_path = inversion_paths[i]
            assert (os.path.exists(w_path)), "Inversion file doesn't exist"
            print(w_path)
            w_vec = self.get_inversion(w_path)
            w_vec = torch.unsqueeze(w_vec, 0)
            W = w_vec if W is None else torch.cat([W, w_vec], dim=0)
            inversion_list.append(inversion_paths[i])
        
        return_list = {'img': I, 'inversion': W, 'raw_desc': raw_desc, "sampleT": sampleT, "framelist": framelist, "index": index, "inversionlist": inversion_list}

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