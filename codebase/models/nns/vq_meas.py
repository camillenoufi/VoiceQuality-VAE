# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim=24, h_dim=16, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim=24, h_dim=16, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, x_dim)
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

class Classifier(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim,
        self.h_dim = h_dim,
        self.net = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, y_dim)
        )

    def forward(self, x):
        return self.net(x)
