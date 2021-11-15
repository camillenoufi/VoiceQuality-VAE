# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='vq_meas', name='vae', z_dim=2, x_dim=24, h_dim=16, beta=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim,
        self.x_dim = x_dim,
        self.h_dim = h_dim,
        self.beta = beta,
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(z_dim, x_dim, h_dim)
        self.dec = nn.Decoder(z_dim, x_dim, h_dim)

        #hyperparams
        self.beta

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        # Note that nelbo = kl + rec
        # Outputs all be scalar
        
        # given x, find params describing normal dist q_phi(z|x)
        qm, qv = self.enc(x)

        #find kl over batch between q(z|x) and p(z), tensor is 1d batch vector
        kl = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)

        # sample z and pass through decoder to get p_theta(xhat|z)
        z = ut.sample_gaussian(qm, qv) #not in log-space
        xhat_logits = self.dec(z)

        # find rec of each sample in batch (summed over dims) using -BCE
        rec = ut.mse(x, xhat_logits)

        # calculate batch averages
        kl = torch.mean(kl, dim=0)
        rec = torch.mean(rec, dim=0)

        # import ipdb; ipdb.set_trace()

        # add KLD and reconstruction loss to get scaled and shifted version of the neg ELBO
        nelbo = rec + torch.mul(self.beta[0],kl)   #true NLL -ln(p(x|z)) = (D/2)*mse + c ->idea: beta will compensate for this shift and scale
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
        
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs all be scalar
        
        # E[ log(mean(exp(logA))) ] == E[ log(mean(A)) ]  
        # RHS is IWAE bound. Use LHS to decompose A into dists we have access to
        # A = p_theta_(x|z^iw) * p_theta_(z^iw) / q_phi_(z^iw|x)
        # logA = log p_theta_(x|z^iw) + log p_theta_(z^iw) - log q_phi_(z^iw|x)
        # logA = log p_theta_(x|z^iw) - ( log p_theta_(z^iw) + log q_phi_(z^iw|x) )
        # logA = log_p_x_given_z - DKL(q||p)

        # given x, find params describing normal dist q_phi(z|x)
        qm, qv = self.enc(x)
        
        # sample z from approx posterior
        qm = ut.duplicate(qm,iw)
        qv = ut.duplicate(qv,iw)
        z = ut.sample_gaussian(qm, qv)

        # get log probs
        log_p_z = ut.log_normal(z, self.z_prior_m, self.z_prior_v)
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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))