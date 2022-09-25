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

#IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
#    '.png', '.PNG'
#]

IMG_EXTENSIONS = ['.png', '.PNG']

def split_sentence_into_words(sentence):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(sentence.lower())

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



class VideoData(data.Dataset):
	def __init__(self, img_root, n_frames_total=15, img_transform=None):
		super(VideoData, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform = transforms.ToTensor()
		self.img_root = img_root
		self.video_list = open(img_root + "train_videos.txt").readlines()
		self.description_list = open(img_root + "train_videos_descriptions.txt").readlines()
		self.multi_description_list = [open(img_root + "train_videos_descriptions.txt").readlines()]
		self.data_paths, self.raw_descriptions = self._load_dataset()
		self.n_of_seqs = len(self.data_paths)                 # number of sequences to train       
		self.seq_len_max = max([len(V) for V in self.data_paths])        
		self.n_frames_total = n_frames_total      # current number of frames to train in a single iteration


	def _load_dataset(self):
		images = []
		descriptions_raw = []
		count = 0 
		descriptions_lists = self.multi_description_list
		for idx, vid_path in enumerate(self.video_list):
			desc_raw = []
			for e in range(len(descriptions_lists)):
				description = descriptions_lists[e][idx]
				raw_desc = description[:-1]
				desc_raw.append(raw_desc)
			paths = []
			description = []
			description_raw = []
			desc_len = []
			raw_desc = raw_desc[:-1]
			fname = vid_path[:-1]
			count = count + 1
			for f in sorted(os.listdir(self.img_root + fname)):
				if is_image_file(f):
					imname = f[:-4]
					paths.append(os.path.join(self.img_root, fname + "/" + f))
					description_raw.append(desc_raw)


			if len(paths) > 0:
				images.append(paths)
				descriptions_raw.append(description_raw)


			print(str(count) + " of " + str(len(self.video_list)))

		return images, descriptions_raw


	def __getitem__(self, index):
		data_paths = self.data_paths[index]
		raw_desc = self.raw_descriptions[index]
		
		num_frames = len(data_paths)    

		sampleT = np.arange(num_frames-1) + 1
		sampleT = np.random.choice(sampleT, 3, replace= False)
		sampleT = np.insert(sampleT, 0, 0, axis=0)
		sampleT = np.sort(sampleT)
		
		# read in images
		I = None
		framelist = []
		for i in sampleT:           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			Ii = self.img_transform(Ii)
			Ii = torch.unsqueeze(Ii, 0)
			I = Ii if I is None else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc, "sampleT" : sampleT, "framelist" : framelist}
		
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoData'
	

class VideoDataTest(data.Dataset):
	def __init__(self, img_root, n_frames_total=15, img_transform=None):
		super(VideoDataTest, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform = transforms.ToTensor()
		self.img_root = img_root
		self.video_list = open(img_root + "test_videos.txt").readlines()[:100]
		self.description_list = open(img_root + "test_videos_descriptions2.txt").readlines()[:100]
		self.multi_description_list = [open(img_root + "test_videos_descriptions2.txt").readlines()[:100]]
		self.data_paths, self.raw_descriptions = self._load_dataset()
		self.n_of_seqs = len(self.data_paths)                 # number of sequences to train       
		self.seq_len_max = max([len(V) for V in self.data_paths])        
		self.n_frames_total = n_frames_total      # current number of frames to train in a single iteration


	def _load_dataset(self):
		images = []
		descriptions_raw = []
		count = 0 
		descriptions_lists = self.multi_description_list
		for idx, vid_path in enumerate(self.video_list):
			desc_raw = []
			for e in range(len(descriptions_lists)):
				description = descriptions_lists[e][idx]
				raw_desc = description[:-1]
				desc_raw.append(raw_desc)
			paths = []
			description = []
			description_raw = []
			desc_len = []
			raw_desc = raw_desc[:-1]
			fname = vid_path[:-1]
			count = count + 1
			for f in sorted(os.listdir(self.img_root + fname)):
				if is_image_file(f):
					imname = f[:-4]
					paths.append(os.path.join(self.img_root, fname + "/" + f))
					description_raw.append(desc_raw)


			if len(paths) > 0:
				images.append(paths)
				descriptions_raw.append(description_raw)


			print(str(count) + " of " + str(len(self.video_list)))

		return images, descriptions_raw


	def __getitem__(self, index):
		data_paths = self.data_paths[index]
		raw_desc = self.raw_descriptions[index]   
		num_frames = len(data_paths)    

		sampleT = np.arange(num_frames-1) + 1
		sampleT = np.random.choice(sampleT, 3, replace= False)
		sampleT = np.insert(sampleT, 0, 0, axis=0)
		sampleT = np.sort(sampleT)
		
		# read in images
		I = None
		framelist = []
		for i in sampleT:           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			Ii = self.img_transform(Ii)
			Ii = torch.unsqueeze(Ii, 0)
			I = Ii if I is None else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc, "sampleT" : sampleT, "framelist" : framelist}
		
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataTest'



class VideoDataFashion(data.Dataset):
	def __init__(self, img_root, n_frames_total, batch_size, img_transform=None, crop=(512, 384), size=(256, 192)):
		super(VideoDataFashion, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform=transforms.Compose([
		                  transforms.CenterCrop(tuple(crop)),
		                  transforms.Resize(tuple(size)),
		                  transforms.ToTensor()
		              ])
		self.img_root = img_root
		self.video_list = open(img_root + "metadata/train_video.txt").readlines()
		self.description_list = open(img_root + "metadata/train_video_descriptions.txt").readlines()
		self.multi_description_list = [open(img_root + "metadata/train_video_descriptions.txt").readlines(), open(img_root + "metadata/train_video_descriptions1.txt").readlines(), open(img_root + "metadata/train_video_descriptions2.txt").readlines(), open(img_root + "metadata/train_video_descriptions3.txt").readlines(), open(img_root + "metadata/train_video_descriptions4.txt").readlines(), open(img_root + "metadata/train_video_descriptions5.txt").readlines()]
		self.data_paths, self.raw_descriptions = self._load_dataset()
		self.n_of_seqs = len(self.data_paths)                 # number of sequences to train       
		self.seq_len_max = max([len(V) for V in self.data_paths])        
		self.n_frames_total = n_frames_total      # current number of frames to train in a single iteration
		self.batch_size = batch_size

	def _load_dataset(self):
		images = []
		descriptions_raw = []
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
			desc_len = []
			raw_desc = raw_desc[:-1]
			fname = vid_path[:-1]
			count = count + 1
			for f in sorted(os.listdir(self.img_root + fname))[:15]: # Question: why limit to 12? Do we know for sure that all videos have at least 25 frames? 
				if is_image_file(f):
					imname = f[:-4]
					#print(imname)
					paths.append(os.path.join(self.img_root, fname + "/" + f))
					description_raw.append(desc_raw)
					#print(len_desc)


			if len(paths) > 0:
				images.append(paths)
				#print(desc_len)
				descriptions_raw.append(description_raw)



		return images, descriptions_raw


	def __getitem__(self, index):
		data_paths = self.data_paths[index]
		rnd_txt = np.random.randint(6)
		raw_desc = self.raw_descriptions[index][:][rnd_txt]
		num_frames = len(data_paths)  

		sampleT = np.arange(num_frames-1) + 1
		local_state = np.random.RandomState(int(index//self.batch_size) + 1)
		np.random.seed(int(index//self.batch_size) + 1)
		sampleT = np.random.choice(sampleT, self.n_frames_total, replace=False)
		sampleT = np.sort(sampleT)
		# print(index, num_frames)
		# read in images
		I = None
		framelist = []
		for i in sampleT:           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			Ii = self.img_transform(Ii)
			Ii = torch.unsqueeze(Ii, 0)
			I = Ii if I is None else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc, "sampleT" : sampleT, "framelist" : framelist, "index": index}
		
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataFashion'

class VideoDataFashionTest(data.Dataset):
	def __init__(self, img_root, n_frames_total, batch_size, img_transform=None, crop=(512, 384), size=(256, 192)):
		super(VideoDataFashionTest, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform=transforms.Compose([
		                  transforms.CenterCrop(tuple(crop)),
		                  transforms.Resize(tuple(size)),
		                  transforms.ToTensor()])
		self.img_root = img_root
		self.video_list = open(img_root + "metadata/test_video.txt").readlines()
		self.description_list = open(img_root + "metadata/test_video_descriptions.txt").readlines()
		self.multi_description_list = [open(img_root + "metadata/test_video_descriptions.txt").readlines()]
		self.data_paths, self.raw_descriptions = self._load_dataset()
		self.n_of_seqs = len(self.data_paths)                 # number of sequences to train       
		self.seq_len_max = max([len(V) for V in self.data_paths])        
		self.n_frames_total = n_frames_total      # current number of frames to train in a single iteration
		self.batch_size = batch_size


	def _load_dataset(self):
		images = []
		descriptions_raw = []
		count = 0 
		descriptions_lists = self.multi_description_list
		for idx, vid_path in enumerate(self.video_list):
			desc_raw = []
			for e in range(1):
				description = descriptions_lists[e][idx]
				raw_desc = description[:-1]
				desc_raw.append(raw_desc)
			paths = []
			description = []
			description_raw = []
			desc_len = []
			raw_desc = raw_desc[:-1]
			fname = vid_path[:-1]
			count = count + 1
			for f in sorted(os.listdir(self.img_root + fname))[:25]:
				if is_image_file(f):
					imname = f[:-4]
					paths.append(os.path.join(self.img_root, fname + "/" + f))
					description_raw.append(desc_raw)


			if len(paths) > 0:
				images.append(paths)
				descriptions_raw.append(description_raw)



		return images, descriptions_raw


	def __getitem__(self, index):
		data_paths = self.data_paths[index]
		raw_desc = self.raw_descriptions[index]
		num_frames = len(data_paths)    

		sampleT = np.arange(num_frames-1) + 1
		np.random.seed(int(index//self.batch_size))
		sampleT = np.random.choice(sampleT, self.n_frames_total, replace=False)
		sampleT = np.sort(sampleT)
		
		# read in images
		I = None
		framelist = []
		for i in sampleT:           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			Ii = self.img_transform(Ii)
			Ii = torch.unsqueeze(Ii, 0)
			I = Ii if I is None else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc, "sampleT" : sampleT, "framelist" : framelist}
		
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataFashionTest'


