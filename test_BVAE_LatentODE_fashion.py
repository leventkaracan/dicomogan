import os
import argparse
import fasttext
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.nn as nn
from model import Encoder,EncoderVideo_LatentODE, Decoder,LatentODEfunc
from data import VideoDataFashionCLIPTest
import torch.utils.data as data
from data import split_sentence_into_words
import random 
from torchdiffeq import odeint
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--model', type=str, required=True,
                    help='pretrained models')
parser.add_argument('--epoch', type=int, required=True,
                    help='pretrained models')
parser.add_argument('--output_root', type=str, required=True,
                    help='root directory of output')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')




parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()


if args.manualSeed is None:
	args.manualSeed = random.randint(1, 10000)
	#args.manualSeed = 6525
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
random.seed(args.manualSeed)
print("Random Seed: ", args.manualSeed)

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True

gpu_ids = []
for str_id in args.gpu_ids.split(','):
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
args.gpu_ids = gpu_ids

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def save_tensor2gif(video_tensor, save_path):
    imTrans = transforms.ToPILImage()
    frames = []
    T = video_tensor.size(0)
    for i in range(T):
        frames.append(imTrans(video_tensor[i].cpu()).convert("RGB")) 
    imageio.mimsave(save_path, frames, fps=15)

def save_batch_images(batch, path, folder_name):
    if not os.path.exists(path + folder_name):
        os.mkdir(path + folder_name)
    bs = batch.size(0)
    for i in range(bs):
       save_image((batch[i].data), path + folder_name + "/" + str(i)+".png")

if __name__ == '__main__':

	if not os.path.exists(args.output_root):
		os.makedirs(args.output_root)
		os.makedirs(os.path.join(args.output_root, "input"))


	print('Loading a dataset...')
	test_data = VideoDataFashionCLIPTest(args.img_root,
		              transforms.Compose([
		                  transforms.CenterCrop((512, 384)),
		                  transforms.Resize((128, 96)),
		                  transforms.ToTensor()
		              ]))

	test_loader = data.DataLoader(test_data,
								   batch_size=1,
								   shuffle=False,
								   num_workers=arg.num_threads)


	print('Loading a generator model...')


	vae_img_transform = transforms.Compose([
								   transforms.ToTensor()
							   ])

	func = LatentODEfunc(4, 20)

	bVAE_enc = EncoderVideo_LatentODE((3,128,96), func, static_latent_dim=12, dynamic_latent_dim=4)
	bVAE_dec = Decoder((3,128,96), latent_dim=16)


	bVAE_enc.eval()
	bVAE_dec.eval()
	func.eval()

	requires_grad(bVAE_enc, False)
	requires_grad(bVAE_dec, False)


	bVAE_enc.load_state_dict(torch.load(args.model + "_V_enc_" + str(args.epoch)))
	bVAE_dec.load_state_dict(torch.load(args.model + "_V_dec_" + str(args.epoch)))



	if not args.no_cuda:

		bVAE_enc.cuda()
		bVAE_dec.cuda()

	label_txt = open(os.path.join(args.output_root, "labels.txt"), "w")



	bVAE_enc.eval()
	bVAE_dec.eval()

	
	model_dir = args.output_root
	dataset = "fashion"

	samples = []
	imTrans = transforms.ToPILImage()
	grids = []
	temp_res = 512
	for i, data in enumerate(test_loader):
                if i>10:
                        break
                print(i)
                video = data[0].cuda()
                bs, T, ch, height, width = video.size()


                sampleT = np.arange(T-1) + 1
                sampleT = np.random.choice(sampleT, 11, replace= False)
                sampleT = np.insert(sampleT, 0, 0, axis=0)
                sampleT = np.sort(sampleT)
                ts = (sampleT + 1)*0.01
                t_tar = (np.sort(np.arange(temp_res)) + 1)*(0.01/(temp_res/T))
                ts = torch.from_numpy(ts).cuda()
                ts = ts - ts[0]
                t_tar = torch.from_numpy(t_tar).cuda()
                t_tar = t_tar - t_tar[0]
                zs, zd = bVAE_enc.test(video[:, sampleT], ts)
                zs_tar, zd_tar = bVAE_enc.test_ode(video[:, sampleT], ts, t_tar)
                np.save(args.output_root + 'ode_pred'+str(i), zd_tar.cpu().numpy())
                z_vid = torch.cat((zs, zd), 1)
                z_vid_tar = torch.cat((zs_tar, zd_tar), 1)
                #print(zd_tar)
                x_recon = bVAE_dec(z_vid)
                x_temp_tar = bVAE_dec(z_vid_tar)
                z_vid_1 = z_vid[0, :].repeat(T,1)
                x_recon_frames = []
                x_recon_tensor = []
                z_vid_l = z_vid_1.clone()
                for l in range(z_vid.size(1)):
                    if l >= zs.size(1):
                        z_vid_l[:, l] = torch.linspace(zd[0,l- zs.size(1)].item(),zd[T-1,l- zs.size(1)].item(),T)
                    else:
                        z_vid_l[:, l] = torch.linspace(-3,3,T)

                    x_recon_l = bVAE_dec(z_vid_l)
                    x_recon_tensor.append(x_recon_l)
                    save_batch_images(x_recon_l, args.output_root, "video"+str(i)+'_latent' + str(l))
                    x_recon_lframes = [imTrans(x_recon_l[zind].cpu()).convert("RGB") for zind in range(z_vid_l.size(0))]
                    x_recon_frames.append(x_recon_lframes)
                    z_vid_l = z_vid_1.clone()

  

                x_recon_tensor = torch.stack(x_recon_tensor)
                input_recon = torch.stack([video.squeeze(0),x_recon])
                grid_frames = []
                for t in range(T):
                    gridTensor = make_grid(torch.cat((input_recon[:, t], x_recon_tensor[:, t]),0), nrow=1)
                    grid_frames.append(gridTensor)
                    save_image((gridTensor.data), './results/real/bvae_ode_test.png')

                save_image(grid_frames, './results/real/bvae_ode_test.png',nrow=12)
                grids.append(torch.stack(grid_frames))


                x_recon_list = []
                input_frames = []
                recon_frames = []
                for zind in range(z_vid.size(0)):
                        input_frames.append(imTrans(video[0, sampleT[zind], ...].cpu()).convert("RGB"))
                        recon_frames.append(imTrans(x_recon[zind].cpu()).convert("RGB"))


                save_image((video[0, :].data), './results/real/video_%d.png' % (i + 1))
                save_batch_images(video.squeeze(0), args.output_root, "video"+str(i))
                imageio.mimsave('./results/real/video_%d.gif' % (i + 1), input_frames, fps=15)
                save_image((x_recon.data), './results/recon/video_rec_%d.png' % (i + 1))
                save_batch_images(x_recon, args.output_root, "video"+str(i)+'_recon')
                save_image(x_temp_tar, args.output_root + "recon_ode_target_"+str(temp_res)+"_"+str(i)+".png", 16)
                save_batch_images(x_temp_tar, args.output_root, "video"+str(i)+'recon_ode_target_'+str(temp_res))
                save_tensor2gif(x_temp_tar, args.output_root + "recon_ode_target_"+str(temp_res)+"_"+str(i)+".gif")
                imageio.mimsave('./results/recon/video_%d.gif' % (i + 1), recon_frames, fps=15)


                for l, lframes in enumerate(x_recon_frames):
                    imageio.mimsave('./results/recon/video0_latent%d.gif' % (l + 1), lframes, fps=15)
	grids = torch.stack(grids)
	#print(grids.size())
	animation_frames = [imTrans( make_grid(grids[:,tind], nrow=grids.size(0)).cpu()).convert("RGB") for tind in range(grids.size(1))]
	grids_tensor = make_grid(grids[:,0], nrow=grids.size(0))
	save_image((grids_tensor.data), './results/real/bvae_ode.png', nrow=grids.size(0))
	imageio.mimsave('./results/real/bvae_ode.gif', animation_frames, fps=15)
