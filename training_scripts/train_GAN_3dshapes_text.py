import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import Image
from dicomogan.model import MultiscaleDiscriminatorPixpixHDAdaINText
from dicomogan.model import Generator
from data.video import VideoData
from dicomogan.losses.loss_lib import GANLoss, VGGLoss
import random
from torchdiffeq import odeint
import clip
from PIL import Image
import math

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
					help='root directory that contains images')
parser.add_argument('--save_filename', type=str, required=True,
					help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
					help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=100,
					help='number of threads for fetching data (default: 100)')
parser.add_argument('--batch_size', type=int, default=64,
					help='batch size (default: 64)')
parser.add_argument('--beta', type=float, default=32, help='beta vae param')

parser.add_argument('--resume', action='store_true', help='resume')
parser.add_argument('--resume_epoch, type=int, default=0
					help='resume epoch(default: 0)')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loss', type=str, default='lsgan')


args = parser.parse_args()

if args.manualSeed is None:
	args.manualSeed = random.randint(1, 10000)
	#args.manualSeed = 6525
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
random.seed(args.manualSeed)
print("Random Seed: ", args.manualSeed)


torch.cuda.set_device(0)



def preprocess_feat(latent_feat):
        bs = int(latent_feat.size(0)/2)
        latent_feat_mismatch = torch.roll(latent_feat, 1, 0)
        latent_splits = torch.split(latent_feat, bs, 0)
        latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], -1, 0), latent_splits[1]), 0)
        return latent_feat_mismatch, latent_feat_relevant


def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag



def linear_annealing(init, fin, step, annealing_steps):
	"""Linear annealing of a parameter."""
	if annealing_steps == 0:
		return fin
	assert fin > init
	delta = fin - init
	annealed = min(init + delta * step / annealing_steps, fin)
	return annealed

if __name__ == '__main__':

	clip_img_transform = transforms.Compose([
transforms.Resize(224, interpolation=Image.BICUBIC),
transforms.CenterCrop(224), 
transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

	print('Loading a dataset...')
	train_data = VideoData(args.img_root,
		              15,
		              transforms.Compose([
		                  transforms.ToTensor()
		              ]))


	train_loader = data.DataLoader(train_data,
								   batch_size=args.batch_size,
								   shuffle=True,
								   num_workers=args.num_threads)


	D = MultiscaleDiscriminatorPixpixHDAdaINText(input_nc=3, ndf=64, norm_layer=nn.InstanceNorm2d)
	G = Generator(fsize=64)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
	requires_grad(clip_model, False)

	epoch_no = 0
	if args.resume:
		print("Resuming from %d. epoch... "%args.resume_epoch)
		G.load_state_dict(torch.load(args.save_filename + "_G_" + str(args.resume_epoch)))
		D.load_state_dict(torch.load(args.save_filename + "_D_" + str(args.resume_epoch)))
		epoch_no = args.resume_epoch


	criterionGAN = GANLoss(use_lsgan=True, target_real_label=1.0)
	criterionFeat = torch.nn.L1Loss()
	criterionVGG = VGGLoss()
	criterionUnsupFactor = torch.nn.MSELoss()

	D.cuda()
	G.cuda()

	g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
	d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
	


	n_train_steps = epoch_no * 4500

	logf = open("./examples_3dshapes/log_max.txt", 'w')
	vae_cond_dim = 3
	ode_dim = 1
	for epoch in range(epoch_no, args.num_epochs):

		# training loop
		avg_D_real_loss = 0
		avg_Dt_real_loss = 0
		avg_D_real_m_loss = 0
		avg_D_real_m_vae_loss = 0
		avg_D_real_m_both_loss = 0
		avg_D_real_v_loss = 0
		avg_Dt_real_m_loss = 0
		avg_D_fake_loss = 0
		avg_D_fake_vae_loss = 0
		avg_D_fake_both_loss = 0
		avg_Dt_fake_loss = 0
		avg_G_fake_loss = 0
		avg_G_fake_temp_loss = 0
		avg_vgg_loss = 0
		avg_vae_loss = 0
		avg_embedvae_loss = 0
		avg_temp_vggflow_loss = 0
		avg_attention_loss = 0
		avg_unsup_loss = 0
		avg_D_real_a_loss = 0
		avg_ganvae_loss = 0
		for i, data in enumerate(train_loader):

			vid = data['img'].cuda()
			input_desc = data['raw_desc']
			#print(*input_desc[0][0])
			#print(len(input_desc))
			text = clip.tokenize([*input_desc[0][0]]).to(device)
			txt_feat = clip_model.encode_text(text).float()

			bs, T, ch, height, width = vid.size()
			sampleT = np.arange(T-1) + 1
			sampleT = np.random.choice(sampleT, 3, replace= False)
			sampleT = np.insert(sampleT, 0, 0, axis=0)
			sampleT = np.sort(sampleT)
			#print(sampleT)
			ts = (sampleT)*0.01
			ts = torch.from_numpy(ts).cuda()
			ts = ts - ts[0]

			vid_norm = vid * 2 - 1
			txt_feat = txt_feat.unsqueeze(0).repeat(4,1,1)
			txt_feat = txt_feat.view(bs * 4, -1)


			requires_grad(G, False)
			requires_grad(D, False)


			# UPDATE DISCRIMINATOR

			requires_grad(G, False)
			requires_grad(D, True)

			D.zero_grad()


			video_sample = vid_norm[:, sampleT[:]]
			video_sample = video_sample.permute(1,0,2,3,4)
			video_sample = video_sample.contiguous().view(bs * 4, ch, height, width)


			# real image with real latent)
			real_logit = D(video_sample, txt_feat)
			real_loss = criterionGAN(real_logit, True)
			avg_D_real_loss += real_loss.data.item()
			real_loss.backward(retain_graph=True)

			# real image with mismatching text
			txt_feat_mismatch,_ = preprocess_feat(txt_feat)
			real_m_logit = D(video_sample, txt_feat_mismatch)
			real_m_loss = 0.5 * criterionGAN(real_m_logit,False)
			avg_D_real_m_loss += real_m_loss.data.item()
			real_m_loss.backward(retain_graph=True)

			# synthesized image with semantically relevant text
			_,txt_feat_relevant = preprocess_feat(txt_feat)
			fake = G(video_sample, txt_feat_relevant)
			fake_logit = D(fake.detach(), txt_feat_relevant)
			fake_loss =  0.5 * criterionGAN(fake_logit, False)
			avg_D_fake_loss += fake_loss.data.item()
			fake_loss.backward(retain_graph=True)

			d_optimizer.step()



			# UPDATE GENERATOR

			requires_grad(G, True)
			requires_grad(D, False)

			G.zero_grad()


			imgsplits = torch.split(video_sample, int((bs * 4)/2), 0)
			img_rel = torch.cat((torch.roll(imgsplits[0], -1, 0), imgsplits[1]), 0)

			_,txt_feat_relevant = preprocess_feat(txt_feat)
			fake = G(video_sample, txt_feat_relevant)
			fake_logit = D(fake,txt_feat_relevant)
			fake_loss = criterionGAN(fake_logit, True)

			#vgg_loss =  criterionVGG(fake, img_rel)
			vgg_loss =  criterionVGG(fake, video_sample)
			avg_G_fake_loss += fake_loss.data.item()
			avg_vgg_loss += vgg_loss.data.item()

			fake_sample = fake.view(4, bs, ch, height, width)
			fake_sample = fake_sample.permute(1,0,2,3,4)

			G_loss = fake_loss + 1.0 * vgg_loss
			G_loss.backward()
			g_optimizer.step()


			if i % 20 == 0:
				print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, '
				      'G_fake: %.4f, VGG: %.4f, TxtEmbed: %.4f, VAE: %.4f, Unsup %.4f, Dt_mis: %.4f, Dt_fake: %.4f, G_fake_temp: %.4f , Dtv :%4f'
					  % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
						 avg_D_real_m_loss/ (i + 1), avg_D_fake_loss/ (i + 1), avg_G_fake_loss/ (i + 1),
						 avg_vgg_loss/ (i + 1), avg_embedvae_loss / (i + 1), avg_vae_loss/(i+1), avg_unsup_loss/(i+1), avg_D_real_v_loss/ (i + 1), avg_Dt_fake_loss/ (i + 1), avg_G_fake_temp_loss/ (i + 1), avg_ganvae_loss / (i + 1)))

			save_image(((fake_sample[0, :].data + 1) * 0.5), './examples_3dshapes/epoch_%d_fake.png' % (epoch + 1))
			save_image((vid[0, sampleT].data), './examples_3dshapes/epoch_%d_real.png' % (epoch + 1))
			torch.save(G.state_dict(), args.save_filename + "_G_" + str(epoch))
			torch.save(D.state_dict(), args.save_filename + "_D_" + str(epoch))

		logf.write('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, '
				      'G_fake: %.4f, VGG: %.4f, TxtEmbed: %.4f, VAE: %.4f, Unsup %.4f, Dt_mis: %.4f, Dt_fake: %.4f, G_fake_temp: %.4f , Dtv :%4f'
					  % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
						 avg_D_real_m_loss / (i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1),
						 avg_vgg_loss / (i + 1), avg_embedvae_loss / (i + 1), avg_vae_loss/(i+1), avg_unsup_loss/(i+1), avg_D_real_v_loss/(i+1), avg_Dt_fake_loss/(i+1), avg_G_fake_temp_loss/(i+1), avg_ganvae_loss / (i + 1)) + "\n")
		logf.flush()

logf.close()
