from utils.helper_functions import visualize_feature_map

import torch
# from models.GAN import Generator, Discriminator
import os
from utils.visualize import visualize_mnist


import numpy as np
from utils.loader import save_model

from torchvision import utils
import torch.nn as nn
from torch.autograd import Variable
import torch
from utils.helper_functions import get_torch_variable

import time as t
import os

class Generator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.latent_dim = 100
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00005)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00005)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

      

def load_model(filename, model):
    model_path = os.path.join(os.getcwd(), filename)
    model.load_state_dict(torch.load(model_path))
    print('model loaded from {}-'.format(model_path))

cuda = True if torch.cuda.is_available() else False
print("is cuda available:", cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

G = Generator()
# D = Discriminator()

# load_model('G.pkl', G.model)
visualize_feature_map(G.model)

# visualize_mnist(G.model[0].numpy(), 1, 'test')
