
import numpy as np
import torch
import torch.nn as nn
import lib.utils as utils
from torchdiffeq import odeint as odeint
from dicomogan.vidode import Encoder, Encoder_z0_ODE_ConvGRU, create_convnet, ODEFunc, DiffeqSolver
from dicomogan.models.swapae_networks.encoder import StyleGAN2ResnetEncoder
# from torchdiffeq import odeint_adjoint as odeint

# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
#         if m.weight is not None:
#             m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]



class EncoderVideo_LatentODE(nn.Module):
    def __init__(self, img_size, static_latent_dim=5, dynamic_latent_dim=1, hid_channels = 32, kernel_size = 4, hidden_dim = 256, use_last_gru_hidden=False,
                 bidirectional_gru=True, num_GRU_layers=1, num_encoder_layers=4, num_conv_layers=2, n_samples=100, 
                 netE_num_downsampling_sp=3, netE_num_downsampling_gl=1, spatial_code_ch=256, global_code_ch=512, sampling_type="Static", n_frames_interpolate=2,
                 n_frames_extrapolate=2):

        # 3dShapes_dataset: static_latent_dim=5, dynamic_latent_dim=1, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        # fashion_dataset: static_latent_dim=12, dynamic_latent_dim=4,, hid_channels = 32, kernel_size = 4, hidden_dim = 256

        super(EncoderVideo_LatentODE, self).__init__()

        self.netE_num_downsampling_sp = netE_num_downsampling_sp
        self.netE_num_downsampling_gl = netE_num_downsampling_gl
        self.static_latent_dim = static_latent_dim
        self.dynamic_latent_dim = dynamic_latent_dim

        self.img_size = img_size
        self.use_last_gru_hidden = use_last_gru_hidden
        self.num_GRU_layers = num_GRU_layers
        self.bidirectional_gru = bidirectional_gru
        self.n_samples = n_samples
        self.sampling_type = sampling_type
        self.n_frames_interp = n_frames_interpolate
        self.n_frames_ext = n_frames_extrapolate

        # channels for encoder, ODE, init decoder
        resize = 2 ** netE_num_downsampling_sp
        base_dim = spatial_code_ch
        input_size = (img_size[0]// resize, img_size[1] // resize)
        ode_dim = base_dim
        self.leakyrelu = nn.LeakyReLU()

        # Shape required to start transpose convs
        self.swapae_encoder = StyleGAN2ResnetEncoder(netE_num_downsampling_sp=netE_num_downsampling_sp, 
                netE_num_downsampling_gl=netE_num_downsampling_gl,
                spatial_code_ch=spatial_code_ch,
                global_code_ch=global_code_ch) 

        
        ##### ODE Encoder
        ode_func_netE = create_convnet(n_inputs=ode_dim, # ode dim should be same as the encoder out dim
                                       n_outputs=base_dim,
                                       n_layers=num_conv_layers,
                                       n_units=base_dim // 2,
                                       dtype=torch.float64)
        
        rec_ode_func = ODEFunc(input_dim=ode_dim,
                               latent_dim=base_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE)
        
        self.z0_diffeq_solver = DiffeqSolver(base_dim,
                                        ode_func=rec_ode_func,
                                        method="euler",
                                        latents=base_dim,
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4)

        # GRUConv
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                 input_dim=spatial_code_ch,
                                                 hidden_dim=spatial_code_ch,
                                                 kernel_size=(3, 3),
                                                 num_layers=num_GRU_layers,
                                                 batch_first=False, # RIP dodge
                                                 bias=True,
                                                 return_all_layers=True,
                                                 z0_diffeq_solver=self.z0_diffeq_solver,
                                                 run_backwards=True,
                                                 dtype=torch.float64)



        ##### ODE Decoder
        ode_func_netD = create_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=num_conv_layers,
                                       n_units=base_dim // 2,
                                       dtype=torch.float64)
        
        gen_ode_func = ODEFunc(input_dim=ode_dim,
                               latent_dim=base_dim,
                               ode_func_net=ode_func_netD)
        
        self.diffeq_solver = DiffeqSolver(base_dim,
                                          gen_ode_func,
                                          'dopri5',
                                          base_dim,
                                          odeint_rtol=1e-3,
                                          odeint_atol=1e-4)

        # self.lin = nn.Linear(spatial_code_ch * (img_size[0] // resize) * (img_size[1] // resize), hidden_dim)
        # self.mu_gen_d = nn.Linear(hidden_dim, self.dynamic_latent_dim)
        self.conv1x1 = nn.Conv2d(spatial_code_ch, self.dynamic_latent_dim, kernel_size=1)

        # self.apply(kaiming_init)

    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps
    
    def video_dynamics(self, x, t, mask_t): # x: B x T x C x H x W , t: (B x T)
        batch_size = x.size(0)
        T = x.size(1)
        xi = x.permute(1, 0, 2, 3, 4).contiguous() # T x B x C x H x W

        # xi = xi.reshape(T*batch_size, x.shape[2], x.shape[3], x.shape[4]) # T*B x C x H x W
        # xi, _ = self.swapae_encoder(xi) # xi: T*B x spatial_code_ch x H' x W'
        # xi = xi.view(T, batch_size, xi.shape[1], xi.shape[2], xi.shape[3]) # T x B x D' x H' x W'

        # TODO: think about normalization should be in which dimention
        all_xi = []
        for sub_x in xi:
            sub_a, _ = self.swapae_encoder(sub_x)
            all_xi.append(sub_a)
            # all_xi_gl.append(sub_b)
        
        xi = torch.stack(all_xi, 0) # T x B x D' x H' x W'
        # xi_gl = torch.stack(all_xi_gl, 0) # T x B x D'


        xi = xi.to(torch.float64)
        ##### ODE encoding
        mask_t = mask_t.unsqueeze(0).repeat(batch_size, 1, 1).to(xi.device).to(torch.float64) # B x T x 1
        zd0, _ = self.encoder_z0(input_tensor=xi.to(torch.float64), time_steps=t, mask=mask_t) # B x spatial_code_ch x H' x W'

        return zd0 
    
    def solve_ode(self, zd0, t):
        # solve for the whole video
        # zd0:  B x spatial_code_ch x H' x W'
        # t: T
        batch_size = zd0.size(0)
        T = t.size(0)

        zdt = self.diffeq_solver(zd0, t) # B x T x spatial_code_ch x H' x W'

        zdt = zdt.permute(1, 0, 2, 3, 4).contiguous().view(batch_size * T, zdt.shape[2], zdt.shape[3], zdt.shape[4]) # T * B x spatial_code_ch x H' x W'
        # zdt = zdt.permute(1, 0, 2, 3, 4).contiguous().view(batch_size * T, -1) # T * B x spatial_code_ch * H' * W'

        zdt = zdt.to(torch.float32)
        # reduce dim to dynamic dim 
        zdt = self.conv1x1(zdt)  # T * B x D x H' x W'
        # zdt = self.leakyrelu(self.lin(zdt))
        # zdt = self.mu_gen_d(zdt)  # T * B x D

        return zdt

    def forward(self, x, t, mask): # x: B x T x C x H x W , t: (B x T)
        zd0 = self.video_dynamics(x, t, mask)
        zdt = self.solve_ode(zd0, t)

        
        # Question: why using hs and not h_max? Isn't redundent? RIP
        # TODO: experiment with non stocastic sampling
        # h_max = torch.max(xi_gl, dim=0)[0] # B x D'
        # zs = self.mu_logvar_gen_s(h_max) # B x 2 * D''
        # zs = zs.unsqueeze(0).repeat(T, 1, 1).view(T * batch_size, -1)

        return zdt