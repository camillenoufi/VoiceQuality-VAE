# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.

        # tuple containing (mean, var) each with shape [N, K, z_dim]
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        # given x, find params describing normal dist q_phi(z|x)
        qm, qv = self.enc(x)

        # sample z from q dist
        z = ut.sample_gaussian(qm, qv)

        # calculate KLD via monte carlo sampling estimation between q and learnable prior pz
        kl = ut.log_normal(z, qm, qv) - ut.log_normal_mixture(z, prior[0], prior[1])
        # print(f'KLD: {kl}')
        
        # pass through decoder to get p_theta(xhat|z)
        xhat_logits = self.dec(z)

        # find rec of each sample in batch (summed over dims) using -BCE
        rec = -ut.log_bernoulli_with_logits(x, xhat_logits)

        # calculate batch averages
        kl = torch.mean(kl, dim=0)
        rec = torch.mean(rec, dim=0)

        # add KLD and negative reconstruction loss to get the neg ELBO
        nelbo = kl + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.

        # E[ log(mean(exp(logA))) ] == E[ log(mean(A)) ]  
        # RHS is IWAE bound. Use LHS to decompose A into dists we have access to
        # A = p_theta_(x|z^iw) * p_theta_(z^iw) / q_phi_(z^iw|x)
        # logA = log p_theta_(x|z^iw) + log p_theta_(z^iw) - log q_phi_(z^iw|x)
        # logA = log p_theta_(x|z^iw) - ( log p_theta_(z^iw) + log q_phi_(z^iw|x) )
        # logA = log_p_x_given_z - DKL(q||p)

        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        # given x, find params describing normal dist q_phi(z|x)
        qm, qv = self.enc(x)
        
        # sample z from approx posterior
        qm = ut.duplicate(qm,iw)
        qv = ut.duplicate(qv,iw)
        z = ut.sample_gaussian(qm, qv)

        # get log probs
        log_p_z = ut.log_normal_mixture(z, prior[0], prior[1])
        log_q_z_given_x = ut.log_normal(z, qm, qv)
        kl_term = log_p_z - log_q_z_given_x

        # pass z through decoder to get p_theta(xhat|z) and then get log prob
        x = ut.duplicate(x,iw)
        log_p_x_given_z = ut.log_bernoulli_with_logits(x, self.dec(z))

        #reshape
        log_p_x_given_z = log_p_x_given_z.reshape(iw, -1)
        kl_term = kl_term.reshape(iw, -1)

        # add together to create inner term logA
        logA = log_p_x_given_z + kl_term

        # import ipdb; ipdb.set_trace()

        # take the expectation over the iw dim and make negative for N-IWAE
        niwae = -torch.mean( ut.log_mean_exp(logA, dim=0), dim=0 )

        # rec and kl for elbo
        rec = -log_p_x_given_z
        kl = -kl_term
        #or just call the neg_elbo function here and return that? unclear in the writeup.


        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
