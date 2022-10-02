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
from dicomogan.modules import MultiscaleDiscriminatorPixpixHDMFMOD
from dicomogan.modules import MappingNetworkVAE, Generator2, EncoderVideo_LatentODE, Decoder, TextEncoder, LatentODEfunc
from dicomogan.data.new_video import VideoDataFashion
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
parser.add_argument('--num_threads', type=int, default=5,
					help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=100,
					help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=2,
					help='batch size (default: 64)')
parser.add_argument('--beta', type=float, default=32, help='beta vae param')

parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch(default: 0)')
parser.add_argument('--resume', action='store_true', help='resume')


parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loss', type=str, default='lsgan')


args = parser.parse_args()

#if not args.no_cuda and not torch.cuda.is_available():
#	print("WARNING: You do not have a CUDA device")
#	args.no_cuda = True

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
print("torch version:", torch.__version__)


# cudnn.benchmark = True


def preprocess_feat(latent_feat):
	bs = int(latent_feat.size(0)/2)
	latent_feat_mismatch = torch.roll(latent_feat, 1, 0)
	latent_splits = torch.split(latent_feat, bs, 0) # split batch in two
	latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], -1, 0), latent_splits[1]), 0) # Question: this swaps the order of the first and second datapoint right?  
	return latent_feat_mismatch, latent_feat_relevant


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

def reparametrize(mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps


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
	clip_img_transform = transforms.Compose([
			transforms.Resize(224, interpolation=Image.BICUBIC),
			transforms.CenterCrop(224), 
			transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

	print('Loading a dataset...')
	train_data = VideoDataFashion(img_root=args.img_root, 
						video_list='data/fashion/fashion_train_videos.txt',
						n_sampled_frames = 4,
						batch_size=args.batch_size,
						crop=[512, 384],
						size=[256, 192],
						)

	train_loader = data.DataLoader(train_data,
								   batch_size=args.batch_size,
								   shuffle=False,
								   num_workers=args.num_threads)


	func = LatentODEfunc(4, 20)
	bVAE_enc = EncoderVideo_LatentODE((3,128,96),func, static_latent_dim=12, dynamic_latent_dim=4)
	bVAE_dec = Decoder((3,128,96), latent_dim=16)
	text_enc = TextEncoder(512, latent_dim=8)

	D = MultiscaleDiscriminatorPixpixHDMFMOD(input_nc=3, ndf=64, norm_layer=nn.InstanceNorm2d)
	G = Generator2(fsize=64)
	mapping = MappingNetworkVAE(input_dim=8, fsize=256)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
	requires_grad(clip_model, False)

	epoch_no = 0
	if args.resume:
		print("Resuming from %d. epoch... "%args.resume_epoch)
		G.load_state_dict(torch.load(args.save_filename + "_G_" + str(args.resume_epoch)))
		D.load_state_dict(torch.load(args.save_filename + "_D_" + str(args.resume_epoch)))
		mapping.load_state_dict(torch.load(args.save_filename + "_M_" + str(args.resume_epoch)))
		bVAE_enc.load_state_dict(torch.load(args.save_filename + "_V_enc_" + str(args.resume_epoch)))
		bVAE_dec.load_state_dict(torch.load(args.save_filename + "_V_dec_" + str(args.resume_epoch)))
		text_enc.load_state_dict(torch.load(args.save_filename + "_V_text_" + str(args.resume_epoch)))
		epoch_no = args.resume_epoch


	criterionGAN = GANLoss(use_lsgan=True, target_real_label=1.0)
	criterionFeat = torch.nn.L1Loss()
	criterionVGG = VGGLoss().cuda()
	criterionUnsupFactor = torch.nn.MSELoss()


	bVAE_enc.cuda()
	text_enc.cuda()
	bVAE_dec.cuda()
	mapping.cuda()
	D.cuda()
	G.cuda()


	vae_enc_optimizer = torch.optim.Adam(bVAE_enc.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	vae_dec_optimizer = torch.optim.Adam(bVAE_dec.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	text_enc_optimizer = torch.optim.Adam(text_enc.parameters(), lr=0.001, betas=(0.9, 0.999))

	g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
	d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
	m_optimizer = torch.optim.Adam(mapping.parameters(), lr=0.000002, betas=(0.5, 0.999))
	


	n_train_steps = epoch_no * 4500
	vae_cond_dim = 8

	logf = open("./examples_fashion/log_max.txt", 'w')

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

			# todo preprocess the embeddings
			text = clip.tokenize(input_desc).to(device)
			txt_feat = clip_model.encode_text(text).float()

			bs, T, ch, height, width = vid.size()

			# sample 4 timestamps include 0
			# Question: is it alright to move sampling to the dataloader? 
			sampleT = data['sampleT'][0]
			# sampleT = np.random.choice(sampleT, 3, replace= False)
			# # TODO: experiment first without ODE 
			# sampleT = np.insert(sampleT, 0, 0, axis=0) # Question: does it have to start with the first frame or it does not matter? We should always start with the first frame. We should have simialr motion in similar timestamps. TODO: fix that in my code 
			# sampleT = np.sort(sampleT)
			#print(sampleT)
			ts = (sampleT)*0.01 # Why normalize ts? ODE training is problematic. Try different values. TODO: research more about normalization for ts
			#print(ts)
			ts = ts.cuda()
			ts = ts - ts[0]

			vid_norm = vid * 2 - 1
			txt_feat = txt_feat.unsqueeze(0).repeat(4,1,1)
			txt_feat = txt_feat.view(bs * 4, -1) # T*B X D




			for g in vae_enc_optimizer.param_groups:
			    vae_enc_lr = g['lr']
			    #print(vae_enc_lr)

	
			requires_grad(bVAE_enc, True)
			requires_grad(text_enc, True)
			requires_grad(bVAE_dec, True)

			requires_grad(G, False)
			requires_grad(D, False)
			requires_grad(mapping, False)	

			bVAE_enc.zero_grad()
			text_enc.zero_grad()
			bVAE_dec.zero_grad()

			video_sample = vid
			video_sample = video_sample.permute(1,0,2,3,4)
			video_sample = video_sample.contiguous().view(bs * 4, ch, height, width)
			# downsample resolution to half
			video_sample = nn.functional.interpolate(video_sample, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
			vid_rs = video_sample.view(4, bs, ch, int(height*0.5),int(width*0.5) )
			vid_rs = vid_rs.permute(1,0,2,3,4)
			zs, zd, mu_logvar_s, mu_logvar_d = bVAE_enc(vid_rs, ts)
			z_vid = torch.cat((zs, zd), 1)

			#print(z_vid.size())

			total_vae_loss = 0.0
			beta_vae_loss = 0.0
			recon_loss = 0.0
			kl_loss = 0.0


			muT, logvarT = text_enc(txt_feat)
			zT = reparametrize(muT, logvarT)

			#print(zT.size())
			#print(z_vid[:, 3:].size())

			# Question: Here we are assuming that the first vae_cond_dim dimentions of zs (z_tr) corresponds to the text-relevant
			# attributes and that replacing these features with features coming from text will still give good reconstruction.
			# However, as zs is not condition on input text, then we should call the first 8 features as the features relavent
			# to the attributes of interest, and that means each gt textual desription should describe these attributes (otherwise, it will be 
			# impossible to reconstruct the image), and thus we are only able to edit these attributes. 
			# However, if we are considering the attributes of interest to be present in all of the images, and that the
			# text of all of the videos describes these attributes, then it is equivalent to saying we are using VAE 
			# with labels to disentangle the set of attribute {A1, ... An}.
			x_reconT = bVAE_dec(torch.cat((zT,z_vid[:, vae_cond_dim:]), 1))

			x_recon = bVAE_dec(z_vid)


			#print(video_sample.size())
			
			recon_loss = reconstruction_loss(video_sample, x_recon, 'bernoulli')
			recon_lossT = reconstruction_loss(video_sample, x_reconT, 'bernoulli')

			kl_loss_d = args.beta * kl_divergence(*mu_logvar_d)
			kl_loss_s = args.beta * kl_divergence(*mu_logvar_s)
			kl_loss = kl_loss_s +  kl_loss_d

			beta_vae_loss = 0.5 * (recon_loss + recon_lossT) +  kl_loss
			beta_vae_loss.backward()
			total_vae_loss += beta_vae_loss.data.item()
			avg_vae_loss += total_vae_loss

			n_train_steps += 1

			vae_enc_optimizer.step()
			vae_dec_optimizer.step()
			text_enc_optimizer.step()


			vid_vis = video_sample.reshape(4, bs, ch, int(0.5 * height), int(0.5 * width))
			rec_vis = x_recon.reshape( 4, bs, ch, int(0.5 * height), int(0.5 * width))
			recT_vis = x_reconT.reshape( 4, bs, ch, int(0.5 * height), int(0.5 * width))
			save_image((vid_vis[:, 0].data), './examples/real/epoch_%d.png' % (epoch + 1))
			save_image((rec_vis[:, 0].data), './examples/recon/epoch_%d.png' % (epoch + 1))
			save_image((recT_vis[:, 0].data), './examples/reconT/epoch_%d.png' % (epoch + 1))

			# UPDATE DISCRIMINATOR
			requires_grad(bVAE_enc, False)
			requires_grad(text_enc, False)
			requires_grad(bVAE_dec, False)

			requires_grad(G, False)
			requires_grad(D, True)
			requires_grad(mapping, False)

			D.zero_grad()

			zs, zd, mu_logvar_s, mu_logvar_d = bVAE_enc(vid_rs, ts)
			z_vid = torch.cat((zs, zd), 1) # T*B x D 

			video_sample = vid_norm
			video_sample = video_sample.permute(1,0,2,3,4)
			video_sample = video_sample.contiguous().view(bs * 4, ch, height, width)


			# real image with matched text and W_cont -> real
			latentw = mapping(z_vid[:,vae_cond_dim:]) # T*B x (D-vae_cond_dim)
			real_logit = D(video_sample, txt_feat, latentw)
			real_loss = criterionGAN(real_logit, True)
			avg_D_real_loss += real_loss.data.item()
			real_loss.backward(retain_graph=True)

			# real image with mismatching text features -> Fake
			latentw = mapping(z_vid[:,vae_cond_dim:]) 
			txt_feat_mismatch,_ = preprocess_feat(txt_feat)
			real_m_logit = D(video_sample, txt_feat_mismatch, latentw)
			real_m_loss = 0.5/3 * criterionGAN(real_m_logit,False)
			avg_D_real_m_loss += real_m_loss.data.item()
			real_m_loss.backward(retain_graph=True)

			# real image with mismatching vae (w_cont) -> Fake
			latentw = mapping(z_vid[:,vae_cond_dim:])
			latentw_mismatch,_ = preprocess_feat(latentw)
			real_m_logit = D(video_sample, txt_feat, latentw_mismatch)
			real_m_loss = 0.5/3 * criterionGAN(real_m_logit,False)
			avg_D_real_m_vae_loss += real_m_loss.data.item()
			real_m_loss.backward(retain_graph=True)

			# real image with mismatching text and vae (W_cont) -> Fake
			latentw = mapping(z_vid[:,vae_cond_dim:])
			latentw_mismatch,_ = preprocess_feat(latentw)
			txt_feat_mismatch,_ = preprocess_feat(txt_feat)
			real_m_logit = D(video_sample, txt_feat_mismatch, latentw_mismatch)
			real_m_loss = 0.5/3 * criterionGAN(real_m_logit,False)
			avg_D_real_m_both_loss += real_m_loss.data.item()
			real_m_loss.backward(retain_graph=True)


			# synthesized image with semantically relevant text -> Fake
			# NOTE: semantically relevant text (real text + fake text) is only paired with synthesized image not real one
			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,txt_feat_relevant = preprocess_feat(txt_feat)
			fake = G(video_sample, txt_feat_relevant, latentw)
			fake_logit = D(fake.detach(), txt_feat_relevant, latentw)
			fake_loss =  0.5/3 * criterionGAN(fake_logit, False)
			avg_D_fake_loss += fake_loss.data.item()
			fake_loss.backward(retain_graph=True)

			# synthesized image with semantically relevant vae -> Fake
			# Question: to obtain semantically relevant VAE, perhaps it would be better to roll frame-wise not batch wise? TODO: experiment with that
			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,latentw_relevant = preprocess_feat(latentw)
			fake = G(video_sample, txt_feat, latentw_relevant)
			fake_logit = D(fake.detach(), txt_feat, latentw_relevant)
			fake_loss =  0.5/3 * criterionGAN(fake_logit, False)
			avg_D_fake_vae_loss += fake_loss.data.item()
			fake_loss.backward(retain_graph=True)

			# synthesized image with semantically relevant text and vae 
			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,latentw_relevant = preprocess_feat(latentw)
			_,txt_feat_relevant = preprocess_feat(txt_feat)
			fake = G(video_sample, txt_feat_relevant, latentw_relevant)
			fake_logit = D(fake.detach(), txt_feat_relevant, latentw_relevant)
			fake_loss =  0.5/3 * criterionGAN(fake_logit, False)
			avg_D_fake_both_loss += fake_loss.data.item()
			fake_loss.backward(retain_graph=True)

			d_optimizer.step()


			unsup_weight = min(1, ((1 - 1e-5) / 2000) * n_train_steps + 1e-5)
			for g in vae_enc_optimizer.param_groups:
			    g['lr'] = vae_enc_lr * unsup_weight * 0.01

			# UPDATE GENERATOR

			requires_grad(bVAE_enc, True)
			requires_grad(bVAE_dec, True)
			requires_grad(text_enc, False)

			requires_grad(mapping, True)
			requires_grad(G, True)
			requires_grad(D, False)


			G.zero_grad()
			mapping.zero_grad()
			bVAE_enc.zero_grad()
			bVAE_dec.zero_grad()

			zs, zd, mu_logvar_s, mu_logvar_d = bVAE_enc(vid_rs, ts)
			z_vid = torch.cat((zs, zd), 1)

			zsplits = torch.split(z_vid, int((bs * 4)/2), 0)
			z_rel = torch.cat((torch.roll(zsplits[0], -1, 0), zsplits[1]), 0)

			imgsplits = torch.split(video_sample, int((bs * 4)/2), 0)
			img_rel = torch.cat((torch.roll(imgsplits[0], -1, 0), imgsplits[1]), 0)

			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,txt_feat_relevant = preprocess_feat(txt_feat)
			fake1 = G(video_sample, txt_feat_relevant, latentw)
			fake_logit = D(fake1,txt_feat_relevant, latentw)
			fake_loss1 = 1.0/3 * criterionGAN(fake_logit, True)

			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,latentw_relevant = preprocess_feat(latentw)
			fake2 = G(video_sample, txt_feat, latentw_relevant)
			fake_logit = D(fake2, txt_feat, latentw_relevant)
			fake_loss2 = 1.0/3 * criterionGAN(fake_logit, True)

			latentw = mapping(z_vid[:,vae_cond_dim:])
			_,latentw_relevant = preprocess_feat(latentw)
			_,txt_feat_relevant = preprocess_feat(txt_feat)

			# print("video sample", video_sample.shape)
			# print("txt_feat_relevant", txt_feat_relevant.shape)
			# print("latentw_relevant", latentw_relevant.shape)

			fake3 = G(video_sample, txt_feat_relevant, latentw_relevant)
			fake_logit = D(fake3, txt_feat_relevant, latentw_relevant)
			fake_loss3 = 1.0/3 * criterionGAN(fake_logit, True)
			
			#vgg_loss =  (criterionVGG(fake3, img_rel) + criterionVGG(fake2, img_rel) + criterionVGG(fake1, img_rel))*(1.0/3.0)
			vgg_loss =  (criterionVGG(fake3, video_sample) + criterionVGG(fake2, video_sample) + criterionVGG(fake1, video_sample))*(1.0/3.0)
			avg_G_fake_loss += (fake_loss1 + fake_loss2 + fake_loss3).data.item()
			avg_vgg_loss += vgg_loss.data.item()




			fake_rs = nn.functional.interpolate(fake1, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
			fake_sample_rs = fake_rs.view(4, bs, ch, int(height*0.5), int(width*0.5))
			fake_sample_rs = fake_sample_rs.permute(1,0,2,3,4)
			fake_sample = fake.view(4, bs, ch, height, width)
			fake_sample = fake_sample.permute(1,0,2,3,4)

			zs_fake, zd_fake, mu_logvar_s_fake, mu_logvar_d_fake = bVAE_enc((fake_sample_rs+1)*0.5, ts)
			z_vid_fake = torch.cat((zs_fake, zd_fake), 1)

			zsplits = torch.split(z_vid, int((bs * 4)/2), 0)
			z_rel = torch.cat((torch.roll(zsplits[0], -1, 0), zsplits[1]), 0)
			unsup_loss = criterionUnsupFactor(z_vid_fake, z_rel)*(1.0/3.0)


			fake_rs = nn.functional.interpolate(fake2, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
			fake_sample_rs = fake_rs.view(4, bs, ch, int(height*0.5), int(width*0.5))
			fake_sample_rs = fake_sample_rs.permute(1,0,2,3,4)
			fake_sample = fake.view(4, bs, ch, height, width)
			fake_sample = fake_sample.permute(1,0,2,3,4)

			zs_fake, zd_fake, mu_logvar_s_fake, mu_logvar_d_fake = bVAE_enc((fake_sample_rs+1)*0.5, ts)
			z_vid_fake = torch.cat((zs_fake, zd_fake), 1)
			zsplits = torch.split(z_vid, int((bs * 4)/2), 0)
			z_rel = torch.cat((torch.roll(zsplits[0], -1, 0), zsplits[1]), 0)
			unsup_loss += criterionUnsupFactor(z_vid_fake, z_rel)*(1.0/3.0)

			fake_rs = nn.functional.interpolate(fake3, scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=True)
			fake_sample_rs = fake_rs.view(4, bs, ch, int(height*0.5), int(width*0.5))
			fake_sample_rs = fake_sample_rs.permute(1,0,2,3,4)
			fake_sample = fake.view(4, bs, ch, height, width)
			fake_sample = fake_sample.permute(1,0,2,3,4)

			zs_fake, zd_fake, mu_logvar_s_fake, mu_logvar_d_fake = bVAE_enc((fake_sample_rs+1)*0.5, ts)
			z_vid_fake = torch.cat((zs_fake, zd_fake), 1)
			zsplits = torch.split(z_vid, int((bs * 4)/2), 0)
			z_rel = torch.cat((torch.roll(zsplits[0], -1, 0), zsplits[1]), 0)
			unsup_loss += criterionUnsupFactor(z_vid_fake, z_rel)*(1.0/3.0)

			avg_unsup_loss += 0.5 * unsup_loss.data.item()

			# TODO: add CLIP loss
			
			G_loss = fake_loss1 + fake_loss2 + fake_loss3 + 1.0 * vgg_loss + 0.5 * unsup_loss
			G_loss.backward()

			g_optimizer.step()
			m_optimizer.step()
			vae_enc_optimizer.step()

			for g in vae_enc_optimizer.param_groups:
			    g['lr'] = vae_enc_lr

			if i % 20 == 0:
				print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f %.4f %.4f , D_fake: %.4f, '
				      'G_fake: %.4f, VGG: %.4f, TxtEmbed: %.4f, VAE: %.4f, Unsup %.4f, Dt_mis: %.4f, Dt_fake: %.4f, G_fake_temp: %.4f , Dtv :%4f'
					  % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
						 avg_D_real_m_loss /(i + 1), avg_D_real_m_vae_loss /(i + 1), avg_D_real_m_both_loss /(i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1),
						 avg_vgg_loss / (i + 1), avg_embedvae_loss / (i + 1), avg_vae_loss/(i+1), avg_unsup_loss/(i+1), avg_D_real_v_loss / (i + 1), avg_Dt_fake_loss/ (i + 1), avg_G_fake_temp_loss/ (i + 1), avg_ganvae_loss/ (i + 1)))



			save_image(((fake.data + 1) * 0.5), './examples_fashion/epoch_%d_fake.png' % (epoch + 1),nrow=bs)
			save_image((((video_sample + 1)*0.5).data), './examples_fashion/epoch_%d_real.png' % (epoch + 1), nrow=bs)
			torch.save(G.state_dict(), args.save_filename + "_G_" + str(epoch))
			torch.save(D.state_dict(), args.save_filename + "_D_" + str(epoch))
			torch.save(mapping.state_dict(), args.save_filename + "_M_" + str(epoch))

			torch.save(bVAE_enc.state_dict(), args.save_filename + "_V_enc_" + str(epoch))
			torch.save(text_enc.state_dict(), args.save_filename + "_V_text_" + str(epoch))
			torch.save(bVAE_dec.state_dict(), args.save_filename + "_V_dec_" + str(epoch))


		logf.write('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f %.4f %.4f, D_fake: %.4f, '
				      'G_fake: %.4f, VGG: %.4f, TxtEmbed: %.4f, VAE: %.4f, Unsup %.4f, Dt_mis: %.4f, Dt_fake: %.4f, G_fake_temp: %.4f , Dtv :%4f'
					  % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss,
						 avg_D_real_m_loss, avg_D_real_m_vae_loss, avg_D_real_m_both_loss, avg_D_fake_loss , avg_G_fake_loss,
						 avg_vgg_loss / (i + 1), avg_embedvae_loss / (i + 1), avg_vae_loss/(i+1), avg_unsup_loss/(i+1), avg_D_real_v_loss, avg_Dt_fake_loss, avg_G_fake_temp_loss, avg_ganvae_loss) + "\n")
		#logf.write("vae enc lr : "+str(vae_enc_lr) + "\n")
		#logf.write("ode func lr : "+str(ode_func_lr) + "\n")
		logf.flush()


logf.close()



# Memeory (r, a, f): 683671552 673091072 10580480
# video sample torch.Size([64, 3, 256, 192])
# txt_feat_relevant torch.Size([64, 512])
# latentw_relevant torch.Size([64, 256])
# Memeory (r, a, f): 29976690688 28615112704 1361577984