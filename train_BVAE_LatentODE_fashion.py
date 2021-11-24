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
from model import Encoder, EncoderVideo_LatentODE, Decoder, ODEFunc2,LatentODEfunc
from data import VideoData, VideoDataFashionCLIP
import random
from torchdiffeq import odeint

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
					help='root directory that contains images')
parser.add_argument('--save_filename', type=str, required=True,
					help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=0,
					help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=300,
					help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=64,
					help='batch size (default: 64)')
parser.add_argument('--beta', type=float, default=32, help='beta vae param')

parser.add_argument('--resume', action='store_true', help='resume')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument(
	'--lr_g', default=1e-4, type=float, help='learning rate of generator'
)
parser.add_argument(
	'--lr_d', default=4e-4, type=float, help='learning rate of discriminator'
)

args = parser.parse_args()

if not torch.cuda.is_available():
	print("WARNING: You do not have a CUDA device")

if args.manualSeed is None:
	args.manualSeed = random.randint(1, 10000)
	#args.manualSeed = 6525
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
random.seed(args.manualSeed)
print("Random Seed: ", args.manualSeed)





# gpu_ids = []
# for str_id in args.gpu_ids.split(','):
# 	id = int(str_id)
# 	if id >= 0:
# 		gpu_ids.append(id)
# args.gpu_ids = gpu_ids
# if len(args.gpu_ids) > 0:
# 	torch.cuda.set_device(args.gpu_ids[0])
# 	torch.cuda.manual_seed_all(args.manualSeed)

torch.cuda.set_device(0)


# cudnn.benchmark = True

def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
        #recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False)

    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False)
    else:
        recon_loss = None

    return recon_loss/batch_size


def kl_divergence(mu, logvar):
	batch_size = mu.size(0)
	latent_dim = mu.size(1)

	latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
	total_kl = latent_kl.sum()
	

	return total_kl




def linear_annealing(init, fin, step, annealing_steps):
	"""Linear annealing of a parameter."""
	if annealing_steps == 0:
		return fin
	assert fin > init
	delta = fin - init
	annealed = min(init + delta * step / annealing_steps, fin)
	return annealed

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr




if __name__ == '__main__':
	img_transform = transforms.Compose([
								   transforms.RandomHorizontalFlip(),
								   transforms.ToTensor()
							   ])

	vae_img_transform = transforms.Compose([
								   transforms.RandomHorizontalFlip(),
								   transforms.ToTensor()
							   ])


	print('Loading a dataset...')
	train_data = VideoDataFashionCLIP(args.img_root,
		              transforms.Compose([
		                  transforms.CenterCrop((512, 384)),
		                  transforms.Resize((128, 96)),

		                  #transforms.RandomHorizontalFlip(),
		                  transforms.ToTensor()
		              ]))

	train_loader = data.DataLoader(train_data,
								   batch_size=args.batch_size,
								   shuffle=True,
								   num_workers=16)


	#func = ODEFunc2(input_dim=1, hidden_dim=20)
	func = LatentODEfunc(8, 20)
	#bVAE_enc = Encoder((3,64,64), func)
	bVAE_enc = EncoderVideo_LatentODE((3,128,96), func, static_latent_dim=12, dynamic_latent_dim=8)
	bVAE_dec = Decoder((3,128,96), latent_dim=20)

	#func = LatentODEfunc(1, 10)



	epoch_no = 0
	model_name = "vae_new3"
	if args.resume:
		bVAE_enc.load_state_dict(torch.load(os.path.join("/home/levent/3dShapes/models/", model_name + "_V_enc_" + str(epoch_no))))
		bVAE_dec.load_state_dict(torch.load(os.path.join("/home/levent/3dShapes/models/", model_name + "_V_dec_" + str(epoch_no))))
	


	#if not args.no_cuda:
	bVAE_enc.cuda()
	bVAE_dec.cuda()

	vae_enc_optimizer = torch.optim.Adam(bVAE_enc.parameters(), lr=0.001, betas=(0.9, 0.999))
	vae_dec_optimizer = torch.optim.Adam(bVAE_dec.parameters(), lr=0.001, betas=(0.9, 0.999))

	#vae_enc_optimizer = torch.optim.Adamax(bVAE_enc.parameters(), lr=0.001, betas=(0.9, 0.999))
	#vae_dec_optimizer = torch.optim.Adamax(bVAE_dec.parameters(), lr=0.001, betas=(0.9, 0.999))
	
	n_train_steps = 0

	for epoch in range(epoch_no, args.num_epochs):
		# d_lr_scheduler.step()
		# g_lr_scheduler.step()

		# training loop
		avg_D_real_loss = 0
		avg_Dt_real_loss = 0
		avg_D_real_m_loss = 0
		avg_Dt_real_m_loss = 0
		avg_D_fake_loss = 0
		avg_Dt_fake_loss = 0
		avg_G_fake_loss = 0
		avg_G_fake_temp_loss = 0
		avg_vgg_loss = 0
		avg_vae_loss = 0
		avg_temp_vggflow_loss = 0
		avg_attention_loss = 0
		avg_unsup_loss = 0


		for i, data in enumerate(train_loader):

						
			#video = data['img'].cuda()
			video = data[1].cuda()
			bs, T, ch, height, width = video.size()
			sampleT = np.arange(T-1) + 1
			sampleT = np.random.choice(sampleT, 3, replace= False)
			sampleT = np.insert(sampleT, 0, 0, axis=0)
			sampleT = np.sort(sampleT)
			#print(sampleT)
			ts = (sampleT)*0.01
			#print(ts)
			ts = torch.from_numpy(ts).cuda()
			ts = ts - ts[0]


			requires_grad(bVAE_enc, True)
			requires_grad(bVAE_dec, True)



			bVAE_enc.zero_grad()
			bVAE_dec.zero_grad()

			#update_learning_rate(vae_enc_optimizer, decay_rate = 0.999, lowest = 0.0001)
			#update_learning_rate(vae_dec_optimizer, decay_rate = 0.999, lowest = 0.0001)		

			zs, zd, mu_logvar_s, mu_logvar_d = bVAE_enc(video[:, sampleT], ts)
			z_vid = torch.cat((zs, zd), 1)

			#print(z_vid.size())

			total_vae_loss = 0.0
			beta_vae_loss = 0.0
			recon_loss = 0.0
			kl_loss = 0.0

			x_recon = bVAE_dec(z_vid)
			video_sample = video[:, sampleT[:]]
			video_sample = video_sample.permute(1,0,2,3,4)
			video_sample = video_sample.contiguous().view(bs * 4, ch, height, width)
			#print(video_sample.size())
			recon_loss = reconstruction_loss(video_sample, x_recon, 'bernoulli')

			kl_loss_d = args.beta * kl_divergence(*mu_logvar_d)
			kl_loss_s = args.beta * kl_divergence(*mu_logvar_s)
			kl_loss = kl_loss_s +  kl_loss_d

			#print(mu_logvar_s[0].size())
			#print(mu_logvar_d[0].size())
			#kl_loss = args.beta * kl_divergence(torch.cat((mu_logvar_s[0],mu_logvar_d[0]), 1),torch.cat((mu_logvar_s[1],mu_logvar_d[1]), 1))
			#print(kl_loss)
			#print(kl_loss_d + kl_loss_s)
			beta_vae_loss = recon_loss +  kl_loss
			beta_vae_loss.backward()
			total_vae_loss += beta_vae_loss
			total_vae_loss += kl_loss_d
			avg_vae_loss += total_vae_loss.data.item()

			n_train_steps += 1

			vae_enc_optimizer.step()
			vae_dec_optimizer.step()


		

			if i % 20 == 0:
				print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, '
				      'G_fake: %.4f, VGG: %.4f, Unsup: %.4f, VAE: %.4f, Dt_real: %.4f, Dt_mis: %.4f, Dt_fake: %.4f, G_fake_temp: %.4f , AttnLoss :%4f'
					  % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
						 avg_D_real_m_loss / (i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1),
						 avg_vgg_loss / (i + 1), avg_unsup_loss / (i + 1), avg_vae_loss/(i+1), avg_Dt_real_loss/(i+1), avg_Dt_real_m_loss/(i+1), avg_Dt_fake_loss/(i+1), avg_G_fake_temp_loss/(i+1), avg_attention_loss / (i + 1)))
			vid_vis = video_sample.reshape(4, bs, ch, height, width)
			rec_vis = x_recon.reshape( 4,bs, ch, height, width)
			save_image((vid_vis[:, 0].data), './examples/real/epoch_%d.png' % (epoch + 1))
			save_image((rec_vis[:, 0].data), './examples/recon/epoch_%d.png' % (epoch + 1))
			#vae = VAETest((3, 64, 64), bVAE_enc, bVAE_dec, 6)
			torch.save(bVAE_enc.state_dict(), args.save_filename + "_V_enc_" + str(epoch))
			torch.save(bVAE_dec.state_dict(), args.save_filename + "_V_dec_" + str(epoch))
			#torch.save(func.state_dict(), args.save_filename + "_V_ode_" + str(epoch))
			#torch.save(vae.state_dict(),  "model.pt")

