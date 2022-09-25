import os
import argparse
import fasttext
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dicomogan.model import Generator2
from dicomogan.model import Encoder, EncoderVideo_LatentODE, Decoder, MappingNetworkVAE, LatentODEfunc
from data.video import VideoDataTest
from data.video import split_sentence_into_words
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import random
import imageio
import clip
from torchdiffeq import odeint

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


if __name__ == '__main__':

	if not os.path.exists(os.path.join(args.output_root, "input")):
		os.makedirs(os.path.join(args.output_root, "input"))
	if not os.path.exists(os.path.join(args.output_root, "output")):
		os.mkdir(os.path.join(args.output_root, "output"))
	if not os.path.exists(os.path.join(args.output_root, "output_edited")):
		os.mkdir(os.path.join(args.output_root, "output_edited"))



	print('Loading test data...')
	test_data = VideoDataTest(args.img_root,15,transforms.ToTensor())
	test_loader = data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)


	print('Loading a generator model...')
	G = Generator2(fsize=64)
	G.eval()
	requires_grad(G, False)


	print('Loading a mapping model...')

	M = MappingNetworkVAE(input_dim=3, fsize=256)
	M.eval()
	requires_grad(M, False)


	func = LatentODEfunc(1, 10)
	bVAE_enc = EncoderVideo_LatentODE((3,64,64),func)
	vae_cond_dim = 3
	bVAE_enc.eval()
	requires_grad(bVAE_enc, False)
	bVAE_dec = Decoder((3,64,64), latent_dim=6)
	bVAE_dec.eval()
	requires_grad(bVAE_dec, False)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
	requires_grad(clip_model, False)


	bVAE_enc.load_state_dict(torch.load(args.model + "_V_enc_" + str(args.epoch)))
	bVAE_dec.load_state_dict(torch.load(args.model + "_V_dec_" + str(args.epoch)))
	G.load_state_dict(torch.load(args.model + "_G_" + str(args.epoch)))
	M.load_state_dict(torch.load(args.model + "_M_" + str(args.epoch)))

	rind = args.model.rfind("/")
	model_name = args.model[rind + 1:]

	if not args.no_cuda:
		bVAE_enc.cuda()
		bVAE_dec.cuda()
		G.cuda()
		M.cuda()

	label_txt = open(os.path.join(args.output_root, "labels.txt"), "w")
	test_descriptions_file = args.img_root + "test_videos_descriptions2.txt"
	descriptions = open(test_descriptions_file).readlines()
	test_descriptions = descriptions
	random.shuffle(descriptions)
	for d in descriptions:
		label_txt.write(d)
	label_txt.close()
	frame_num = 15
	original_text = []
	imTrans = transforms.ToPILImage()
	for i, data in enumerate(test_loader):

		if i>5:
			break

		print("video : %d" % i)
		vid = data['img'].cuda()
		input_desc = data['raw_desc']
		#print(*input_desc[0][0])
		original_text.append(*input_desc[0][0])
		#print(len(input_desc))
		text = clip.tokenize([*input_desc[0][0]]).to(device)
		txt_feat_orig = clip_model.encode_text(text).float()

		video = data['img'].cuda()
		video_norm = video * 2 - 1

		bs, T, ch, height, width = video.size()
		sampleT = np.arange(T-1) + 1
		sampleT = np.random.choice(sampleT, T - 1, replace= False)
		sampleT = np.insert(sampleT, 0, 0, axis=0)
		sampleT = np.sort(sampleT)
		ts = (sampleT + 1)*0.01
		ts = torch.from_numpy(ts).cuda()
		ts = ts - ts[0]
		zs, zd = bVAE_enc.test(video[:, sampleT], ts)
		z_vid = torch.cat((zs, zd), 1)
		#print(z_vid.size())

		video_sample = video_norm[:, sampleT[:]]
		video_sample = video_sample.permute(1,0,2,3,4)
		video_sample = video_sample.contiguous().view(bs * frame_num, ch, height, width)



		txt = test_descriptions[i]
		txt_target = descriptions[i]

		txt = txt.replace('\n', '')
		txt_target = txt.replace('\n', '')
		text_target = clip.tokenize([txt_target]).to(device)
		txt_feat_target = clip_model.encode_text(text_target).float()

		txt_feat_orig = txt_feat_orig.unsqueeze(0).repeat(frame_num,1,1)
		txt_feat_orig = txt_feat_orig.view(bs * frame_num, -1)
		txt_feat_target = txt_feat_target.unsqueeze(0).repeat(frame_num,1,1)
		txt_feat_target = txt_feat_target.view(bs * frame_num, -1)

		latentw_orig = M(z_vid[:,3:])
		latentw_target = M(z_vid[:,3:])
		fake_orig = G(video_sample, txt_feat_orig, latentw_orig)
		fake_target = G(video_sample, txt_feat_target, latentw_target)
		save_image(((video_sample.data)), args.output_root + '/video_%d_inp.png' % (i + 1))
		save_image(((fake_orig.data + 1) * 0.5), args.output_root + 'video_%d_rec.png' % (i + 1))
		save_image(((fake_target.data + 1) * 0.5), args.output_root + 'video_%d_tar.png' % (i + 1))

		input_frames = [imTrans(((video_sample[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_inp.gif', input_frames, fps=15)

		recon_frames = [imTrans(((fake_orig[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_rec.gif', recon_frames, fps=15)

		edited_frames = [imTrans(((fake_target[ind] + 1) * 0.5).cpu()).convert("RGB") for ind in range(video_sample.size(0))]
		imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_tar.gif', edited_frames, fps=15)

		#zvid0 = z_vid[0].unsqueeze(0).repeat(frame_num, 1, 1).view(bs * frame_num, -1)
		zvid0 = z_vid[0, vae_cond_dim:].unsqueeze(0).repeat(frame_num, 1, 1).view(bs * frame_num, -1)
		zvid0c = z_vid[0, :vae_cond_dim].unsqueeze(0).repeat(frame_num, 1, 1).view(bs * frame_num, -1)
		first_frame = video_sample[0]
		first_frame = first_frame.unsqueeze(0).repeat(frame_num, 1, 1 ,1 ,1)
		first_frame = first_frame.view(bs * frame_num, ch, height, width)
		#print(first_frame.size())
		#print(zvid0.size())

		z_vid_l = zvid0.clone()
		print(z_vid_l.size(1))
		for l in range(0, z_vid_l.size(1)):
			if l >= 2:
				z_vid_l[:, l] = torch.linspace(zd[0,l-2].item(),zd[T-1,l-2].item(),T)
			else:
				z_vid_l[:, l] = torch.linspace(-1,1,15)
			#print(z_vid_l[:,3:])
			latentw_target = M(z_vid_l)
			input_video = first_frame
			if l == 2:
				input_video = bVAE_dec(torch.cat((zvid0c, z_vid_l),1))
			fake_target = G(input_video, txt_feat_target, latentw_target)
			edited_frames = [imTrans(((fake_target[tind] + 1) * 0.5).cpu()).convert("RGB") for tind in range(z_vid_l.size(0))]
			save_image((fake_target + 1) * 0.5, args.output_root + 'video_' + str(i + 1) + '_vae'+str(l + 1)+'.png')
			imageio.mimsave(args.output_root + 'video_' + str(i + 1) + '_vae' + str(l + 1) + '.gif', edited_frames, fps=15)
			z_vid_l = zvid0.clone()


	html = '<html>\n'
	html += '<body>\n'
	html += '<h1>Manipulated Images</h1>\n'
	html += '<table border="1px solid gray" style="width=100%">\n'
	html += '<tr><td></td><td><b>Description</b></td><td><b>GIF</b></td><td  colspan="8"><b>Frames</b></td></tr>\n'

	for i in range(5):
		html += '<tr>\n'
		html += "<td><b> Input Original : </b></td>\n" 
		html += '<td>\n'
		html +=  original_text[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_inp.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_inp.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'

		html += '<tr>\n'
		html += "<td><b> Reconstructed :</b></td>\n" 
		html += '<td>\n'
		html +=  original_text[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_rec.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_rec.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'

		html += '<tr>\n'
		html += "<td><b> Edited by Text :</b></td>\n" 
		html += '<td>\n'
		html +=  descriptions[i]
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_tar.gif width="128px">'
		html += '</td>\n'
		html += '<td>\n'
		html += '<img src=''./video_'+str(i + 1)+'_tar.png width="1024px">'
		html += '</td>\n'
		html += '</tr>\n'
		for l in range(z_vid_l.size(1)):
			html += '<tr>\n'
			html += "<td><b> Edited by VAE Dim "+ str(l + 1) +":</b></td>\n" 
			html += '<td>\n'
			html +=  descriptions[i]
			html += '</td>\n'
			html += '<td>\n'
			html += '<img src=''./video_'+str(i + 1)+'_vae'+str(l + 1)+'.gif width="128px">'
			html += '</td>\n'
			html += '<td>\n'
			html += '<img src=''./video_'+str(i + 1)+'_vae'+str(l + 1)+'.png width="1024px">'
			html += '</td>\n'
			html += '</tr>\n'


		html += '<tr height="28"></tr>\n'

	html += '</table>\n'
	html += '</body>\n'
	html += '</html>\n'

	f = open(args.output_root + "result.html",'w+')
	f.write(html)
	f.close()

		

	print('Done. The results were saved in %s' % args.output_root)
