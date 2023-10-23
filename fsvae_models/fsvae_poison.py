import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv


# def hook_backward(grad):
#     # print(module)  # 为了区分模块
#     print('grad_output: ', grad.shape, grad.max(), grad.min(), grad.mean())


class FSVAEPoison(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.half_latent_dim = int(self.latent_dim / 2)
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            self.latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(self.latent_dim),
                                            spike=LIFSpike())

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(self.half_latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, x):
        # encode
        x = self.encoder(x)  # (N, C, H, W, T)

        # get mu and generate sampled_z
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N, C*H*W, T)
        latent_x = self.before_latent_layer(x)        # (N, latent_dim, T)
        rate_mu = latent_x[:, :self.half_latent_dim, :]
        log_var = latent_x[:, self.half_latent_dim:, :]
        # rate_mu = torch.mean(latent_x, dim=-1)       # (N, latent_dim)
        sampled_z, full_sampled_z = self.reparameterize(rate_mu.mean(dim=-1))   # (N, latent_dim, T), (N, latent_dim, K, T)

        # decoder input
        result = self.decoder_input(sampled_z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)

        # decode
        result = self.decoder(result)

        # reconstruction
        result = self.final_layer(result)  # (N,C,H,W,T)
        x_recon = torch.tanh(self.membrane_output_layer(result))
        return x_recon, rate_mu, log_var, sampled_z

    def reparameterize(self, mu):
        lif_node = MyLIFSpike()
        x = torch.rand([mu.shape[0], mu.shape[1], self.n_steps]).to(mu.device)
        sample_z = lif_node([x, mu])
        return sample_z, sample_z

    def sample(self, batch_size, device=torch.device("cuda:0")):
        rate_mu = torch.rand(batch_size, self.half_latent_dim).to(device)
        sampled_z, full_sampled_z = self.reparameterize(rate_mu)      # (N, latent_dim, T)

        # decoder input
        result = self.decoder_input(sampled_z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)

        # decode
        result = self.decoder(result)

        # reconstruction
        result = self.final_layer(result)  # (N,C,H,W,T)
        sampled_x = torch.tanh(self.membrane_output_layer(result))
        return sampled_x, sampled_z

    def _inverse_gumbel_cdf(self, y, gumbel_mu, gumbel_beta):
        return gumbel_mu - gumbel_beta * torch.log(-torch.log(y))

    def _gumbel_softmax_sampling(self, mu, gumbel_mu=0, gumbel_beta=1, tau=0.1):
        """
        mu : (N, latent_dim, K) tensor. Assume we need to sample a N*latent_dim*K tensor, each row is an independent r.v.
        """
        shape_mu = mu.shape
        y = torch.rand(shape_mu).to(mu.device) + 1e-25  # ensure all y is positive.
        g = self._inverse_gumbel_cdf(y, gumbel_mu, gumbel_beta).to(mu.device)
        x = torch.log(mu + 1e-7) + g  # samples follow Gumbel distribution.
        # using softmax to generate one_hot vector:
        x = x / tau
        x = F.softmax(x, dim=-1)  # now, the x approximates a one_hot vector.
        return x

    def loss_function_poison_kl(self, input_img, recons_img, mu, sampled_z):
        """
        q(z|x) = mu^z*(1-mu)^(1-z)
        mu: (N, latent_dim)
        sampled_z: (N, latent_dim, T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)

        kl_loss_part_1 = (1 - mu) * (torch.log((1 - mu) / (1/2.50663) + 1e-7))
        kl_loss_part_2 = mu * (torch.log(mu / (0.60653/2.50663) + 1e-7))
        kl_loss = kl_loss_part_1 + kl_loss_part_2
        kl_loss = kl_loss.sum(dim=-1).mean()

        # mu = mu.unsqueeze(-1).repeat(1, 1, self.n_steps)   # (N, latent_dim, T)
        # kl_loss_part_1 = (mu**sampled_z) * ((1-mu)**(1-sampled_z))
        # kl_loss_part_1 = torch.cumprod(kl_loss_part_1, dim=1)[:, -1, :].mean(axis=0)    # (N, T)
        #
        # kl_loss = 0
        # for t in range(self.n_steps):
        #     kl_loss += kl_loss_part_1[t] * (F.binary_cross_entropy_with_logits(mu, sampled_z) + self.latent_dim * torch.log(torch.tensor([2])).to(mu.device))

        loss = recons_loss + 1e-2 * kl_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': kl_loss}

    def loss_function_mu_var_kld(self, input_img, recons_img, mu, log_var):
        # mu: [N, latent_size, T]
        # log_var: [N, latent_size, T]
        N, _, T = mu.shape

        recons_loss = F.mse_loss(recons_img, input_img)

        kld_loss = 0
        for t in range(T):
            kld_element = mu[..., t].pow(2).add_(log_var[..., t].exp()).mul_(-1).add_(1).add_(log_var[..., t])
            kld_t = torch.sum(kld_element).mul_(-0.5) / N
            kld_loss += kld_t
        kld_loss /= T

        loss = recons_loss + 1e-4 * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': kld_loss}

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class FSVAELargePoison(FSVAEPoison):
    def __init__(self):
        super(FSVAEPoison, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()
