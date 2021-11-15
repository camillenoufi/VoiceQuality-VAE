# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

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
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        y_logits = self.cls(x)
        y_logprob = F.log_softmax(y_logits, dim=1) #log_q
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim) , q(y|x)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y]) #[batch x y_dim, y_dim]
        x = ut.duplicate(x, self.y_dim) #[batch x y_dim, x_dim]

        # ask about the above block.... why are we making labels that might never exist? seems like we are saying all labels are equallty likely
        
        # KL_y
        log_p = -np.log(self.y_dim)
                            #q     log_q   
        kl_y = ut.kl_cat(y_prob, y_logprob, log_p) #log_p = log of uniform categorical prior 1/y_dim

        #KL_z
        qm, qv = self.enc(x, y) #[batch x y_dim, z_dim]
        kl_z = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v) #[batch x y_dim]
        # import ipdb; ipdb.set_trace()
        kl_z = kl_z.reshape(self.y_dim, -1) * y_prob.T #[y_dim, batch]
        # kl_z = kl_z.reshape(-1, self.y_dim) * y_prob #[batch, y_dim]
        kl_z = kl_z.sum(0) #sum over y, [batch]

        #Rec
        z = ut.sample_gaussian(qm, qv) # sample z from q(z|x,y), shape is [batch x y_dim, z_dim]

        xhats = self.dec(z, y)
        log_p_x_given_zy = ut.log_bernoulli_with_logits(x, xhats) #log( p(x|(z,y)) ), size is [batch x y_dim]
        rec = -log_p_x_given_zy.reshape(self.y_dim, -1) * y_prob.T # [batch] 
        rec = rec.sum(0) #sum over y

        #reshape and take batch means
        kl_y = kl_y.mean()
        kl_z = kl_z.mean()
        rec = rec.mean()

        #Negative ELBO
        nelbo = kl_y + kl_z + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
