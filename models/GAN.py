
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

      

def train_1_epoch(generator: Generator, 
                    discriminator: Discriminator, 
                    cuda=True, n_critic=10, 
                    data=None, 
                    batch_size=64, 
                    debug=False, 
                    n_epochs=1000, 
                    lambda_term=10,
                    g_iter=-1,
                    id=-1,
                    root=''):
    # attrs = vars(self)
    latent_dim = 100
    clip_value = 0.01

    t_begin = t.time()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for d_iter in range(n_critic):
        images = data.__next__()
        if (images.size()[0] != batch_size):
            continue
        discriminator.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(0)
        real_images = Variable(images.type(Tensor))
        fake_images = generator(z).detach()

        loss_D = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))
        loss_D.backward()
        discriminator.optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        if debug:
            print(f'  Discriminator iteration: {d_iter+1}/{n_critic}, d_loss: {loss_D}')
    # train G
    generator.optimizer.zero_grad()
    z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(0)
    gen_imgs = generator(z)
    loss_G = -torch.mean(discriminator(gen_imgs))
    loss_G.backward()

    generator.optimizer.step()
    print(f'Generator iteration: {g_iter}/{n_epochs}, g_loss: {loss_G}')
    if (g_iter-1) % 100 == 0:
        if not os.path.exists('{}/training_result_images/'.format(root)):
            os.makedirs('{}/training_result_images/'.format(root))
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(0)
        samples = generator(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, '{}/training_result_images/img_generatori_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))
        
        
    time = t.time() - t_begin
    print("Time {}".format(time))
    save_model(generator, discriminator, id, root)