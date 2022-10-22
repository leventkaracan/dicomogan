
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
                 netE_num_downsampling_sp=3, netE_num_downsampling_gl=1, spatial_code_ch=256, global_code_ch=512, sampling_type="Static"):

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

        # channels for encoder, ODE, init decoder
        resize = 2 ** netE_num_downsampling_sp
        base_dim = spatial_code_ch
        input_size = (img_size[0]// resize, img_size[1] // resize)
        ode_dim = base_dim

        # Shape required to start transpose convs
        self.swapae_encoder = StyleGAN2ResnetEncoder(netE_num_downsampling_sp=netE_num_downsampling_sp, 
                netE_num_downsampling_gl=netE_num_downsampling_gl,
                spatial_code_ch=spatial_code_ch,
                global_code_ch=global_code_ch) # TODO: configure this

        
        ##### ODE Encoder
        ode_func_netE = create_convnet(n_inputs=ode_dim, # ode dim should be same as the encoder out dim
                                       n_outputs=base_dim,
                                       n_layers=num_conv_layers,
                                       n_units=base_dim // 2)
        
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
                                                 dtype=torch.cuda.FloatTensor, # TODO: fix later
                                                 batch_first=False, # RIP dodge
                                                 bias=True,
                                                 return_all_layers=True,
                                                 z0_diffeq_solver=self.z0_diffeq_solver,
                                                 run_backwards=True)



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

        self.lin = nn.Linear(spatial_code_ch * (img_size[0] // resize) * (img_size[1] // resize), hidden_dim)
        self.mu_gen_d = nn.Linear(hidden_dim, self.dynamic_latent_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen_s = nn.Linear(global_code_ch, self.static_latent_dim)

        # self.apply(kaiming_init)

    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps

    def forward(self, x, t): # x: B x T x C x H x W , t: (B x T)
        batch_size = x.size(0)
        T = x.size(1)

        xi = x.permute(1, 0, 2, 3, 4).contiguous() # T x B x C x H x W
        xi = xi.reshape(T*batch_size, x.shape[2], x.shape[3], x.shape[4]) # T*B x C x H x W
        xi, xi_gl = self.swapae_encoder(xi) # xi: T*B x spatial_code_ch x H' x W', xi_gl: T*B x global_code_ch
        xi = xi.view(T, batch_size, xi.shape[1], xi.shape[2], xi.shape[3]) # T x B x D' x H' x W'
        xi_gl = xi_gl.view(T, batch_size, -1)  # T x B x D'


        # all_xi, all_xi_gl = [], []
        # for sub_x in xi:
        #     sub_a, sub_b = self.swapae_encoder(sub_x)
        #     all_xi.append(sub_a)
        #     all_xi_gl.append(sub_b)
        
        # xi = torch.stack(all_xi, 0) # T x B x D' x H' x W'
        # xi_gl = torch.stack(all_xi_gl, 0) # T x B x D'




        ##### ODE encoding
        mask = torch.zeros(T, 1) # TODO: make mask
        # TODO: differenciate between interp and exterp when sampling
        if self.sampling_type == "Interpolate":
            inds = np.random.choice(np.arange(1, T-1), min(self.n_samples, T) - 4, replace=False)
            mask[inds, :] = 1
            mask[0, :] = 1
            mask[T-1, :] = 1
        elif self.sampling_type == "Extrapolate":
            #inds = np.random.choice(T-1, min(self.n_samples, T) - 1, replace=False)
            inds = np.arange(0, T-1)
            mask[inds, :] = 1
            #mask[0, :] = 1
        elif self.sampling_type == "Static":
            mask = torch.ones(T, 1)
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1).to(xi.device) # B x T x 1
        zd0, _ = self.encoder_z0(input_tensor=xi, time_steps=t, mask=mask) # B x spatial_code_ch x H' x W'

        # solve for the whole video
        zd0 = zd0.to(torch.float64)
        zdt = self.diffeq_solver(zd0, t) # B x T x spatial_code_ch x H' x W'
        zdt = zdt.to(torch.float32) # B x T x spatial_code_ch x H' x W'
        zdt = zdt.permute(1, 0, 2, 3, 4).contiguous().view(batch_size * T, -1) # T * B x spatial_code_ch * H' * W'

        # reduce dim to dynamic dim 
        zdt = torch.relu(self.lin(zdt))
        zdt = self.mu_gen_d(zdt)  # T * B x D

        # Question: why using hs and not h_max? Isn't redundent? RIP
        # TODO: experiment with non stocastic sampling
        h_max = torch.max(xi_gl, dim=0)[0] # B x D'
        zs = self.mu_logvar_gen_s(h_max) # B x 2 * D''
        zs = zs.unsqueeze(0).repeat(T, 1, 1).view(T * batch_size, -1)

        return zs, zdt, (None, None), (None, None)