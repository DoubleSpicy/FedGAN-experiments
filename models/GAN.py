
import numpy as np
from utils.loader import save_model

from torchvision import utils
import torch.nn as nn
from torch.autograd import Variable
import torch
from utils.helper_functions import get_torch_variable

import time as t
import os

import torch.distributed as dist

from utils.lossLogger import lossLogger, FIDLogger
from utils.fid.fid_score import compute_FID
import shutil

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
        # print('img_flat device:', img_flat.get_device())
        validity = self.model(img_flat)
        return validity

def update(generator: Generator, 
                    discriminator: Discriminator, 
                    cuda=True, n_critic=10, 
                    data=None, 
                    batch_size=64, 
                    debug=False, 
                    n_epochs=1000, 
                    lambda_term=10,
                    g_iter=-1,
                    id=-1,
                    root='',
                    size=0,
                    D_only=False,
                    loss_logger: lossLogger = None):
    # attrs = vars(self)
    latent_dim = 100
    clip_value = 0.01

    t_begin = t.time()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for p in discriminator.parameters():
        p.requires_grad = True

    for d_iter in range(n_critic):
        images = data.__next__()
        if (images.size()[0] != batch_size):
            continue
        discriminator.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(id % size)
        real_images = Variable(images.type(Tensor)).to(id % size)
        fake_images = generator(z).detach().to(id % size)

        d_loss_real = -torch.mean(discriminator(real_images))
        d_loss_fake =  torch.mean(discriminator(fake_images))
        loss_D = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))
        loss_D.backward()
        discriminator.optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        if debug:
            print(f'  Device:{id%size} | ID:{id} | g_iter:{g_iter} | d_iter: {d_iter+1}/{n_critic}, d_loss: {loss_D}')

    for p in discriminator.parameters():
        p.requires_grad = False   

    if not D_only:
        # train G
        generator.optimizer.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(id % size)
        gen_imgs = generator(z).cuda(id % size)
        # print(gen_imgs.size())
        g_loss = -torch.mean(discriminator(gen_imgs)).cuda(id % size)
        g_loss.backward()

        generator.optimizer.step()
        print(f'Device:{id%size} | ID:{id} | g_iter:{g_iter}/{n_epochs} | g_loss: {g_loss}')
        loss_logger.concat([d_loss_real, d_loss_fake, g_loss])
        if (g_iter) % 100 == 0:
            if not os.path.exists('{}/training_result_images/'.format(root)):
                os.makedirs('{}/training_result_images/'.format(root))
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim)))).cuda(id % size)
            samples = generator(z).cuda(id % size)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = utils.make_grid(samples)
            utils.save_image(grid, '{}/training_result_images/img_generatori_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))
        
        
    time = t.time() - t_begin
    print("Time {}".format(time))
    save_model(generator, discriminator, id, root)


# def save_sample(generator: Generator, cuda_index: int, root: str, g_iter: str):
#     cuda = True if torch.cuda.is_available() else False
#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     size = dist.get_world_size()
#     if not os.path.exists('{}/training_result_images/'.format(root)):
#         os.makedirs('{}/training_result_images/'.format(root))
#         z = Variable(Tensor(np.random.normal(0, 1, (64, 100)))).cuda(id % size)
#         samples = generator(z).cuda(id % size)
#         samples = samples.mul(0.5).add(0.5)
#         samples = samples.data.cpu()[:64]
#         grid = utils.make_grid(samples)
#         utils.save_image(grid, '{}/training_result_images/img_generatori_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))

def save_sample(generator: Generator, cuda_index: int, root: str, g_iter: int):
    # Denormalize images and save them in grid 8x8
    samples = generate_images(generator, 64, cuda_index)
    grid = utils.make_grid(samples)
    utils.save_image(grid, '{}/training_result_images/afterAvg_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), dist.get_rank()))

def generate_images(generator: Generator, batch_size = 64, cuda_index = 0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for param in generator.parameters():
        param.requires_grad = False
    z = z = Variable(Tensor(np.random.normal(0, 1, (64, 100)))).cuda(cuda_index)
    samples = generator(z).to(cuda_index)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data[:64]
    for param in generator.parameters():
        param.requires_grad = True
    return samples

def calculate_FID(root: str, generator: Generator, discriminator: Discriminator, device: int, rank: int, share_D: bool):
    path = os.path.join(root, f'temp{device}')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    size = dist.get_world_size()
    for param in generator.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False
    for i in range(157): # make 10k of samples 
        images = generate_images(generator, 64, device)
        samples = torch.unbind(images, dim=0)
        for j in range(len(samples)):
            utils.save_image(samples[j], path + "/" + str(i*64+j).zfill(6) + '.png')
    dist.barrier()
    score = []
    for j in range(size):
        # score.append(j)
        fid_value, diff, tr_sigma1, tr_sigma2, tr_convmean, tr, frobenious, sigma1 = compute_FID([path, os.path.join(root, f'data{j}.npz')], rank=rank)
        score.extend([fid_value, diff, tr_sigma1, tr_sigma2, tr_convmean, tr, frobenious, sigma1])
        # score.append(calculate_FID(root=root, generator=generator, discriminator=discriminator, device=rank, npz_path=os.path.join(root, f'data{j}.npz'), rank=rank))
        # print(f'{rank} vs data{j}: {score[j]}')
    # score.append(calculate_FID(root=root, generator=generator, discriminator=discriminator, device=rank, npz_path=os.path.join(root, f'dataAll.npz'), rank=rank))
    fid_value, diff, tr_sigma1, tr_sigma2, tr_convmean, tr, frobenious, sigma1 = compute_FID([path, os.path.join(root, f'dataAll.npz')], rank=rank)
    score.extend([fid_value, diff, tr_sigma1, tr_sigma2, tr_convmean, tr, frobenious, sigma1])
    dist.barrier()
    shutil.rmtree(path)
    for param in generator.parameters():
        param.requires_grad = True
    for param in discriminator.parameters():
        param.requires_grad = True
    return score