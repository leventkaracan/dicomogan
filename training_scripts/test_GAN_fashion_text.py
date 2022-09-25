import os
import argparse
import fasttext
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dicomogan.model import Generator, VisualSemanticEmbedding
from data.video import VideoDataFashionTest
from data.video import split_sentence_into_words
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import random
import imageio
import clip
from torchdiffeq import odeint
import fasttext
from nltk.tokenize import RegexpTokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--img_root', type=str, required=True,
					help='root directory that contains images')

parser.add_argument('--output_root', type=str, required=True,
					help='root directory of output')
parser.add_argument('--no_cuda', action='store_true',
					help='do not use cuda')
parser.add_argument('--max_nwords', type=int, default=10,
					help='maximum number of words (default: 50)')

parser.add_argument('--model', type=str, required=True,
                    help='pretrained model')
parser.add_argument('--epoch', type=int, required=True,
                    help='pretrained epoch')
parser.add_argument('--embed_ndim', type=int, default=300,
					help='dimension of embedded vector (default: 300)')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()


if not args.no_cuda and not torch.cuda.is_available():
	print('Warning: cuda is not available on this machine.')
	args.no_cuda = True

gpu_ids = []
for str_id in args.gpu_ids.split(','):
	id = int(str_id)
	if id >= 0:
		gpu_ids.append(id)
args.gpu_ids = gpu_ids


img_transform = transforms.Compose([transforms.ToTensor()])

random.seed(99)
torch.manual_seed(99)
np.random.seed(99)
torch.cuda.manual_seed(99)



def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag

def split_sentence_into_words(sentence):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(sentence.lower())


if __name__ == '__main__':

	if not os.path.exists(os.path.join(args.output_root, "input")):
		os.makedirs(os.path.join(args.output_root, "input"))
	if not os.path.exists(os.path.join(args.output_root, "output")):
		os.mkdir(os.path.join(args.output_root, "output"))
	if not os.path.exists(os.path.join(args.output_root, "output_edited")):
		os.mkdir(os.path.join(args.output_root, "output_edited"))




	print('Loading test data...')
	test_data = VideoDataFashionTest(args.img_root, img_transform=transforms.Compose([
		                  transforms.CenterCrop((512, 384)),
		                  transforms.Resize((256, 192)),
		                  #transforms.RandomHorizontalFlip(),
		                  transforms.ToTensor()
		              ]))
	test_loader = data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)


	print('Loading a generator model...')
	G = Generator(fsize=64)
	G.eval()
	requires_grad(G, False)


	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
	requires_grad(clip_model, False)

	G.load_state_dict(torch.load(args.model + "_G_" + str(args.epoch)))


	rind = args.model.rfind("/")
	model_name = args.model[rind + 1:]

	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
	requires_grad(clip_model, False)


	if not args.no_cuda:
		G.cuda()

	label_txt = open(os.path.join(args.output_root, "labels.txt"), "w")
	test_descriptions_file = args.img_root+"metadata/test_descriptions.txt"
	descriptions = open(test_descriptions_file).readlines()
	test_descriptions = descriptions.copy()
	random.shuffle(descriptions)
	for d in descriptions:
		label_txt.write(d)
	label_txt.close()
	frame_num = 12
	original_text = []
	imTrans = transforms.ToPILImage()

	for i, data in enumerate(test_loader):

		if i>10:
			break

		print("video : %d" % i)
		vid = data['img'].cuda()
		input_desc = data['raw_desc']



		#print(text)
		video = data['img'].cuda()
		video_norm = video * 2 - 1
		#print(video)
		bs, T, ch, height, width = video.size()
		sampleT = np.arange(T-1) + 1
		sampleT = np.random.choice(sampleT, frame_num - 1, replace= False)
		sampleT = np.insert(sampleT, 0, 0, axis=0)
		sampleT = np.sort(sampleT)
		ts = (sampleT + 1)*0.01
		ts = torch.from_numpy(ts).cuda()
		ts = ts - ts[0]
		
		print(*input_desc[0][0])
		original_text.append(*input_desc[0][0])

		text = clip.tokenize([*input_desc[0][0]]).to(device)
		txt_feat_orig = clip_model.encode_text(text).float()
		txt_feat_orig = txt_feat_orig.unsqueeze(0).repeat(frame_num,1,1)
		txt_feat_orig = txt_feat_orig.view(bs * frame_num, -1)

		video_sample_norm = video_norm[:, sampleT[:]]
		video_sample = video[:, sampleT[:]]
		video_sample_norm = video_sample_norm.permute(1,0,2,3,4)
		video_sample = video_sample.permute(1,0,2,3,4)
		video_sample_norm = video_sample_norm.contiguous().view(bs * frame_num, ch, height, width)
		video_sample = video_sample.contiguous().view(bs * frame_num, ch, height, width)


		#txt = test_descriptions[100]
		txt_target = descriptions[i]
		#txt_target = "Red Pullovers Made from viscose-nylon blend Round neckline Pearl detail Long sleeves Regular fit"
		#txt = txt.replace('\n', '')
		txt_target = txt_target.replace('\n', '')
		text_target = clip.tokenize([txt_target]).to(device)
		txt_feat_target = clip_model.encode_text(text_target).float()


		txt_feat_target = txt_feat_target.unsqueeze(0).repeat(frame_num,1,1)
		txt_feat_target = txt_feat_target.view(bs * frame_num, -1)


		#print(latentw_orig- latentw_target)
		#print()

		fake_orig = G(video_sample_norm, txt_feat_orig)
		fake_target = G(video_sample_norm, txt_feat_target)
		save_image(((video_sample_norm.data + 1) * 0.5), args.output_root + '/video_%d_inp.png' % (i + 1))
		save_image(((fake_orig.data + 1) * 0.5), args.output_root + 'video_%d_rec.png' % (i + 1))
		save_image(((fake_target.data + 1) * 0.5), args.output_root + 'video_%d_tar.png' % (i + 1))

		input_frames = [imTrans(((video_sample[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_inp.gif', input_frames, fps=15)

		recon_frames = [imTrans(((fake_orig[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_rec.gif', recon_frames, fps=15)

		edited_frames = [imTrans(((fake_target[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_tar.gif', edited_frames, fps=15)



	html = '<html>\n'
	html += '<body>\n'
	html += '<h1>Manipulated Images</h1>\n'
	html += '<table border="1px solid gray" style="width=100%">\n'
	html += '<tr><td></td><td><b>Description</b></td><td><b>GIF</b></td><td  colspan="8"><b>Frames</b></td></tr>\n'

	for i in range(10):
		html += '<tr>\n'
		html += "<td><b> Input Original : </b></td>\n" 
		html += '<td>\n'
		html +=  original_text[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_inp.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_inp.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'

		html += '<tr>\n'
		html += "<td><b> Reconstructed :</b></td>\n" 
		html += '<td>\n'
		html +=  original_text[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_rec.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_rec.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'

		html += '<tr>\n'
		html += "<td><b> Edited by Text :</b></td>\n" 
		html += '<td>\n'
		html +=  descriptions[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_tar.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src='+ args.output_root + 'video_'+str(i + 1)+'_tar.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'


		html += '<tr height="28"></tr>\n'

	html += '</table>\n'
	html += '</body>\n'
	html += '</html>\n'

	f = open("./result.html",'w+')
	f.write(html)
	f.close()

		

	print('Done. The results were saved in %s' % args.output_root)
