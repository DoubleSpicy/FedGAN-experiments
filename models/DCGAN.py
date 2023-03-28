import torch.nn as nn
import torch
from utils.fid.fid_score import compute_FID
from utils.lossLogger import lossLogger, FIDLogger

from utils.helper_functions import get_torch_variable
from torchvision import utils

import os 
import shutil

from torch.autograd import Variable
import torch.distributed as dist

import numpy as np
# # Number of workers for dataloader
# workers = 2

# # Batch size during training
# batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.


# Generator Code

image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.g_optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.d_optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        
    def forward(self, input):
        return self.model(input)
# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
                    loss_logger: lossLogger = None
                    ):

    real_label, fake_label = 1, 0
    cuda_index = id % size
    criterion = nn.BCELoss().to(cuda_index)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    discriminator.zero_grad()
    # Format batch
    real_cpu = data.__next__()
    real_cpu = get_torch_variable(real_cpu, True, cuda_index)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=cuda_index)
    # Forward pass real batch through D
    output = discriminator(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.rand((batch_size, 100, 1, 1)).to(cuda_index)
    noise = get_torch_variable(noise, True, cuda_index)
    # Generate fake image batch with G
    fake = generator(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = discriminator(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    discriminator.d_optimizer.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    generator.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = discriminator(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    generator.g_optimizer.step()

    # Output training stats
    if g_iter % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (g_iter, g_iter, g_iter, len(real_cpu),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save Losses for plotting later
    # G_losses.append(errG.item())
    # D_losses.append(errD.item())

    # Check how the generator is doing by saving G's output on fixed_noise
    # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
    #     with torch.no_grad():
    #         fake = generator(fixed_noise).detach().cpu()
    #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # iters += 1

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
    z = torch.rand((batch_size, 100, 1, 1)).to(cuda_index)
    z = get_torch_variable(z, True, cuda_index)
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