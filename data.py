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
		n_frames_total = len(data_paths)    

		# read in images
		I = 0
		W = 0
		F = 0
		Wl = []
		clip_size = self.n_frames_total
		start_frame = np.random.randint(n_frames_total - clip_size + 1)
		framelist = []
		rawdescs = []
		rnd = random.random()
		for i in range(len(data_paths)):           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			Ii = self.img_transform(Ii)
			Ii = torch. unsqueeze(Ii, 0)
			I = Ii if i == start_frame else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc}
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
		n_frames_total = len(data_paths)    

		# read in images
		I = 0
		W = 0
		F = 0
		Wl = []
		#n_frames_total = 16
		clip_size = self.n_frames_total
		start_frame = np.random.randint(n_frames_total - clip_size + 1)
		framelist = []
		rawdescs = []

		rnd = random.random()
		for i in range(len(data_paths)):           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			"""
			if rnd < 0.5:
				Ii = func.hflip(Ii)
			"""
			Ii = self.img_transform(Ii)
			Ii = torch. unsqueeze(Ii, 0)
			I = Ii if i == start_frame else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc}
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataTest'



class VideoDataFashion(data.Dataset):
	def __init__(self, img_root, n_frames_total=15, img_transform=None):
		super(VideoDataFashion, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform = transforms.ToTensor()
		self.img_root = img_root
		self.video_list = open(img_root + "metadata/train_video.txt").readlines()
		self.description_list = open(img_root + "metadata/train_video_descriptions.txt").readlines()
		self.multi_description_list = [open(img_root + "metadata/train_video_descriptions.txt").readlines(), open(img_root + "metadata/train_video_descriptions1.txt").readlines(), open(img_root + "metadata/train_video_descriptions2.txt").readlines(), open(img_root + "metadata/train_video_descriptions3.txt").readlines(), open(img_root + "metadata/train_video_descriptions4.txt").readlines(), open(img_root + "metadata/train_video_descriptions5.txt").readlines()]
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
			for f in sorted(os.listdir(self.img_root + fname))[:12]:
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


			print(str(count) + " of " + str(len(self.video_list)))

		return images, descriptions_raw


	def __getitem__(self, index):
		data_paths = self.data_paths[index]
		rnd_txt = np.random.randint(6)
		raw_desc = self.raw_descriptions[index][:][rnd_txt]
		n_frames_total = len(data_paths)    

		# read in images
		I = 0
		W = 0
		F = 0
		Wl = []


		clip_size = 12
		T = np.arange(clip_size)
		framelist = []
		rawdescs = []
		start_frame = 0
		rnd = random.random()
		for i in list(T):           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			"""
			if rnd < 0.5:
				Ii = func.hflip(Ii)
			"""
			Ii = self.img_transform(Ii)
			Ii = torch. unsqueeze(Ii, 0)
			I = Ii if i == start_frame else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc}
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataFashion'


class VideoDataFashionTest(data.Dataset):
	def __init__(self, img_root, n_frames_total=15, img_transform=None):
		super(VideoDataFashionTest, self).__init__()
		self.img_transform = img_transform
		if self.img_transform is None:
			self.img_transform = transforms.ToTensor()
		self.img_root = img_root
		self.video_list = open(img_root + "metadata/test_video.txt").readlines()
		self.description_list = open(img_root + "metadata/test_video_descriptions.txt").readlines()
		self.multi_description_list = [open(img_root + "metadata/test_video_descriptions.txt").readlines()]
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
			for f in sorted(os.listdir(self.img_root + fname))[:12]:
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
		n_frames_total = len(data_paths)    

		# read in images
		I = 0
		W = 0
		F = 0
		Wl = []

		clip_size = 12
		T = np.arange(clip_size)

		framelist = []
		rawdescs = []
		start_frame = 0
		rnd = random.random()
		for i in list(T):           
			I_path = data_paths[i]
			Ii = self.get_image(I_path)
			"""
			if rnd < 0.5:
				Ii = func.hflip(Ii)
			"""
			Ii = self.img_transform(Ii)
			Ii = torch. unsqueeze(Ii, 0)
			I = Ii if i == start_frame else torch.cat([I, Ii], dim=0)
			framelist.append(data_paths[i]) 
		return_list = {'img': I, 'raw_desc' : raw_desc}
		return return_list

	def get_image(self, img_path):
		img = Image.open(img_path).convert('RGB')        
		return img
	def __len__(self):
		return len(self.data_paths)

	def name(self):
		return 'VideoDataFashionTest'


