import torch.nn as nn
import torch
import time as t
import os 
from torchvision import utils
from torch import autograd
from utils.loader import save_model

from utils.helper_functions import get_torch_variable
from torch.autograd import Variable
import torch.distributed as dist


class Generator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.model = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
            # output of main module --> Image (Cx32x32)

        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = 1e-4
        self.g_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, img_shape):
        channels = img_shape[0]
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.model = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

        # Optimizers
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = 1e-4
        self.d_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

    def forward(self, x):
        x = self.model(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.model(x)
        return x.view(-1, 1024*4*4)


def train_1_epoch_D(generator: Generator, 
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
                    size=0):
    t_begin = t.time()

    

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if cuda:
        cuda_index=id % size
        one = one.cuda(cuda_index)
        mone = mone.cuda(cuda_index)

    for p in discriminator.parameters():
        p.requires_grad = True

    d_loss_real = 0
    d_loss_fake = 0
    Wasserstein_D = 0
    # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
    for d_iter in range(n_critic):
        discriminator.zero_grad()

        images = data.__next__()
        # Check for batch to have full batch_size
        if (images.size()[0] != batch_size):
            continue

        z = torch.rand((batch_size, 100, 1, 1)).to(cuda_index)

        images, z = get_torch_variable(images, True, cuda_index), get_torch_variable(z, True, cuda_index)

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        d_loss_real = discriminator(images)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)

        # Train with fake images
        z = get_torch_variable(torch.randn(batch_size, 100, 1, 1)).to(cuda_index)

        fake_images = generator(z).to(cuda_index)
        d_loss_fake = discriminator(fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)

        # Train with gradient penalty
        gradient_penalty = calculate_gradient_penalty(images.data, fake_images.data, 
                                                    discriminator,
                                                    batch_size, 
                                                    cuda, 
                                                    cuda_index, 
                                                    lambda_term)
        gradient_penalty.backward()


        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        discriminator.d_optimizer.step()
        if debug:
            print(f'  Discriminator iteration: {d_iter+1}/{n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')



    t_end = t.time()
    print('Time of training-{}'.format((t_end - t_begin)))
    # Save the trained parameters
    save_model(generator, discriminator, id, root)

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
                    root='',
                    size=0):
    t_begin = t.time()

    

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if cuda:
        cuda_index=id % size
        one = one.cuda(cuda_index)
        mone = mone.cuda(cuda_index)

    for p in discriminator.parameters():
        p.requires_grad = True

    d_loss_real = 0
    d_loss_fake = 0
    Wasserstein_D = 0
    # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
    for d_iter in range(n_critic):
        discriminator.zero_grad()

        images = data.__next__()
        # Check for batch to have full batch_size
        if (images.size()[0] != batch_size):
            continue

        z = torch.rand((batch_size, 100, 1, 1)).to(cuda_index)

        images, z = get_torch_variable(images, True, cuda_index), get_torch_variable(z, True, cuda_index)

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        d_loss_real = discriminator(images)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)

        # Train with fake images
        z = get_torch_variable(torch.randn(batch_size, 100, 1, 1), True, cuda_index)

        fake_images = generator(z).to(cuda_index)
        d_loss_fake = discriminator(fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)

        # Train with gradient penalty
        gradient_penalty = calculate_gradient_penalty(images.data, fake_images.data, 
                                                    discriminator,
                                                    batch_size, 
                                                    cuda, 
                                                    cuda_index, 
                                                    lambda_term)
        gradient_penalty.backward()


        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        discriminator.d_optimizer.step()
        if debug:
            print(f'  Discriminator iteration: {d_iter+1}/{n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

    # Generator update
    for p in discriminator.parameters():
        p.requires_grad = False  # to avoid computation

    generator.zero_grad()
    # train generator
    # compute loss with fake images
    z = get_torch_variable(torch.randn(batch_size, 100, 1, 1), True, cuda_index)
    fake_images = generator(z).to(cuda_index)
    g_loss = discriminator(fake_images)
    g_loss = g_loss.mean()
    g_loss.backward(mone)
    g_cost = -g_loss
    generator.g_optimizer.step()
    print(f'Device:{dist.get_rank()}: | ID:{id} | Generator iteration: {g_iter}/{n_epochs}, g_loss: {g_loss}')
    # Saving model and sampling images every 1000th generator iterations
    if (g_iter) % 100 == 0:
        if not os.path.exists('{}/training_result_images/'.format(root)):
            os.makedirs('{}/training_result_images/'.format(root))

        # Denormalize images and save them in grid 8x8
        z = get_torch_variable(torch.randn(800, 100, 1, 1), True, cuda_index)
        samples = generator(z).to(cuda_index)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, '{}/training_result_images/img_generatori_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))

        # Testing
        time = t.time() - t_begin
        #print("Real Inception score: {}".format(inception_score))
        print(f"Device:{dist.get_rank()}: | ID:{id} | Generator iter: {g_iter}")
        print("Time {}".format(time))



    t_end = t.time()
    print(f'Device:{dist.get_rank()}: | ID:{id} | Time of training-{(t_end - t_begin)}')
    # Save the trained parameters
    save_model(generator, discriminator, id, root)






def calculate_gradient_penalty(real_images, 
                            fake_images, 
                            discriminator, 
                            batch_size, 
                            cuda, 
                            cuda_index, 
                            lambda_term):
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    if cuda:
        eta = eta.cuda(cuda_index)
    else:
        eta = eta

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    if cuda:
        interpolated = interpolated.cuda(cuda_index)
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).cuda(cuda_index) if cuda else torch.ones(
                                prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty


def save_sample(generator: Generator, cuda_index: int, root: str, g_iter: int):
    # Denormalize images and save them in grid 8x8
    z = get_torch_variable(torch.randn(800, 100, 1, 1), True, cuda_index)
    samples = generator(z).to(cuda_index)
    print(samples)
    # samples = samples.mul(0.5).add(0.5)
    # samples = samples.data.cpu()[:64]
    # grid = utils.make_grid(samples)
    # utils.save_image(grid, '{}/training_result_images/afterAvg_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))