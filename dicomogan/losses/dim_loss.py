
"""Module for networks used for computing MI.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)

class MIFCNet(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units, bn =False):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)


        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        """

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: network output.

        """


        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)

        return h


class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units,):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """
        res = self.block_nonlinear(x) 
        res += self.linear_shortcut(x)
        h = self.block_ln(res)
        return h

class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim
        return

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x




def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y

def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def generator_loss(q_samples, measure, loss_type=None):
    """Computes the loss for the generator of a GAN.

    Args:
        q_samples: fake samples.
        measure: Measure to compute loss for.
        loss_type: Type of loss: basic `minimax` or `non-saturating`.

    """
    if not loss_type or loss_type == 'minimax':
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    else:
        raise NotImplementedError(
            'Generator loss type `{}` not supported. '
            'Supported: [None, non-saturating, boundary-seek]')


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.

    Note that vectors should be sent as 1x1.

    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.

    Returns:
        torch.Tensor: Loss.

    # '''
    # print("[fenchel_dual_loss] l shape:", l.shape) # B x units x n_local (64)
    # print("[fenchel_dual_loss] m shape:", m.shape) # B x units x n_multis (1)
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units) # B * 64 x units

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units) # B * 1 x units

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t()) # matrix multipication between (B * 1, units) and (units, B * 64) -> (B * 1, B * 64)
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1) # B x B x 1 x 64

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).type_as(l)
    n_mask = 1 - mask # only one in the diagonal of B x B matrix

    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)

    # Mask positive and negative terms for positive and negative parts of loss
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos

    return loss


def infonce_loss(l, m):
    '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].type_as(l)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss


def donsker_varadhan_loss(l, m):
    '''

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).type_as(l)
    n_mask = 1 - mask

    # Positive term is just the average of the diagonal.
    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    # Negative term is the log sum exp of the off-diagonal terms. Mask out the positive.
    u -= 10 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos

    return loss


class DIMLoss(nn.Module):
    def __init__(self, n_units=2048, embed1_dim=256, embed2_dim=512, measure='JSD', mode='fd', scale=1.0, act_penalty=0, loss_start=0):
        super().__init__()
        self.measure = measure
        self.mode = mode
        self.n_units = n_units
        self.scale = scale
        self.act_penalty = act_penalty
        self.img_embed_dim = embed1_dim
        self.cap_embed_dim = embed2_dim
        self.loss_start = loss_start


        # MI estimators
        local_mi = MI1x1ConvNet(embed1_dim, n_units)
        summary_mi = MI1x1ConvNet(embed2_dim, n_units)
        self.mi_networks = nn.ModuleList([local_mi, summary_mi])

    def compute_dim_loss(self, l_enc, m_enc):
        '''Computes DIM loss.

        Args:
            l_enc: Local feature map encoding.
            m_enc: Multiple globals feature map encoding.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        Returns:
            torch.Tensor: Loss.

        '''

        if self.mode == 'fd':
            loss = fenchel_dual_loss(l_enc, m_enc, measure=self.measure)
        elif self.mode == 'nce':
            loss = infonce_loss(l_enc, m_enc)
        elif self.mode == 'dv':
            loss = donsker_varadhan_loss(l_enc, m_enc)
        else:
            raise NotImplementedError(self.mode)
            
        return loss

    def forward(self, embed1, embed2):
        """[summary]

        Args:
            img_embed ([type]): B x D
            embed2 ([type]): B x D

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        bs = embed1.shape[0]
        
        # pass to MI networks
        c_units = self.mi_networks[1](embed2[:, :, None, None].float()).view(bs, self.n_units, -1) # (B, units, 1, 1)
        
        local_units = self.mi_networks[0](embed1[:, :, None, None].float()).view(bs, self.n_units, -1) # (B, n_units, 1)
        
        mi_loss = self.compute_dim_loss(local_units, c_units)
        mi_loss /= c_units.shape[0]
        act_loss = (c_units ** 2).sum(2).mean().mean()
        tot_loss = self.scale * mi_loss + self.act_penalty * act_loss

        return tot_loss

    def configure_optimizers(self):
        return list(self.mi_networks[0].parameters()) +  list(self.mi_networks[1].parameters())