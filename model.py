import sys, os

from pandas import get_option
sys.path.append(os.getcwd())

# from image_processor import CustomColorChange


import argparse
# import enum
from typing import OrderedDict
import numpy as np
# import math
import random

'''
this is the image processor on the dataset so that each clinet gets MNIST images of different colors
'''

# import enum
import os
from typing import OrderedDict
import numpy as np
# import math
# import sys
# import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, utils
from torch.autograd import Variable
from torch import autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil

import time as t
import matplotlib.pyplot as plt
os.makedirs("../../data/mnist", exist_ok=True)

dir = './dataset_transform'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

if os.path.exists('./images'):
    shutil.rmtree('./images')
os.makedirs('./images')

import cv2 as cv
from PIL import Image

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

def cal_split_lengths(length, n):
    # divide a length into n approx. equal parts
    div = length // n
    ans = [div for i in range(n)]
    ans[-1] += length - div*n
    print(ans)
    return ans

def visualize_length(set):
    print("length of set: ", end='')
    print(set)
    for i in enumerate(set):
        print(len(i), end=' ')
    print('')

def prob_true(limit):
    return random.uniform(0, 1) >= limit

def BGR2PIL(img):
    # BGR opencv -> PIL
    cv.cvtColor(img, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

class CustomColorChange():
    ''' Change background color
        https://pytorch.org/vision/stable/transforms.html'''
    def __init__(self, colors:tuple = None, all_random=False):
        if colors == None:
            self.colors =  (random.randint(0, 255), random.randint(0, 180), random.randint(0, 255))
        else:
            self.colors = colors
        self.all_random = all_random
        self.first_time = True
        return
    
    def __call__(self, imgPIL):
        imgRGB = np.array(imgPIL)
        imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)

        # imgRGB = cv.cvtColor(imgRGB, cv.COLOR_BGR2BGR)
        imgHSV = cv.cvtColor(imgBGR, cv.COLOR_RGB2HSV)
        if debug:
            cv.imwrite('testimgHSV.jpg',cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR))

        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 125])
        mask = cv.inRange(imgHSV, lower, upper)
        cv.imwrite('testimgMask.jpg', cv.cvtColor(mask, cv.COLOR_GRAY2BGR))

        color_mask = np.zeros((32, 32, 3), np.uint8)
        # color_mask[:] = (125, 180, 255) # red/random
        if self.all_random:
            color_mask[:] = (random.randint(0, 255), random.randint(0, 180), random.randint(0, 255))
        else:
            color_mask[:] = self.colors # red/random
            # print(self.colors)
        if debug:
            cv.imwrite('testimgColorMask.jpg', cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        color_mask = cv.bitwise_and(color_mask, color_mask, mask=mask)
        imgBGR = cv.add(imgBGR, cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        if debug:
            cv.imwrite('testimgMasked.jpg', cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB))
        pil_img = BGR2PIL(imgBGR)
        if debug:
            pil_img.save('testimgPIL.jpg')
        # print("called!!")
        if self.first_time:
            print(self.colors)
            self.first_time = False
        return pil_img



def nparray_to_PIL(nparr):
    # convert for opencv use
    nparr *= 255
    return Image.fromarray(nparr.astype(np.uint8))


os.makedirs("images", exist_ok=True)

# no share D, 100 epochs
parser = argparse.ArgumentParser()
n_epochs = 1000
batch_size = 64
lr = 0.00005
n_cpu = 8
latent_dim = 100
img_size = 28
channels = 3
n_critic = 10
clip_value = 0.01
sample_interval = 400
client_cnt = 4
share_D = False
debug = True # global debug variable
load_G, load_D = True, True

# dataset = datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform = transforms.Compose([
#             transforms.Resize(32),
#             transforms.Grayscale(3),
#             CustomColorChange(all_random=True),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, ), (0.5, )),
#         ]))
# print(len(dataset))
# trainset = random_split(dataset, cal_split_lengths(len(dataset), client_cnt), generator=torch.Generator().manual_seed(42))
# split n sets for the train and test set


trainset = []
for i in range(client_cnt):
    trainset.append(
        datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            CustomColorChange(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
    ))
    idx = [prob_true(0.99) for i in trainset[i].targets]
    trainset[i].targets = trainset[i].targets[idx]
    trainset[i].data = trainset[i].data[idx]
    print("trainset[i] length:", len(trainset[i]))





print('rubbish pytorch')
trainloader = list()
for i in range(0, client_cnt):
    trainloader.append(torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
                                            shuffle=True, drop_last=True))
# [torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
#                                             shuffle=False) for i in range(client_cnt)]
# testloader = [torch.utils.data.DataLoader(testset[i], batch_size=batch_size,
#                                             shuffle=False, drop_last=True) for i in range(client_cnt)]

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False
print("is cuda available:", cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
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
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
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


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class Server():
    # for doing model averaging after updating local ones
    def __init__(self, client_list:list()):
        self.clients = client_list
        self.generator = Generator(3)
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = 1e-4
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        if cuda:
            self.generator.cuda()
        if load_G:
            self.load_model("generator_pretrain.pkl")
    def load_model(self, G_model_filename):
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.generator.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))

    def average(self):
        print("averaging models on %d clients..."%(len(self.clients)))
        averaged_dis = OrderedDict()
        
        for it, client in enumerate(self.clients):
            dis_params = client.discriminator.state_dict()
            coefficient = 1/len(self.clients)
            if share_D:
                for idx, val in dis_params.items():
                    if idx not in averaged_dis:
                        averaged_dis[idx] = 0
                    averaged_dis[idx] += val * coefficient

        print("updating models for all clients")
        for idx, client in enumerate(self.clients):
            if share_D:
                self.clients[idx].discriminator.load_state_dict(dis_params)
        if debug:
            for idx, client in enumerate(self.clients):
                print("Comparing discriminator [{}]".format(idx))
                self.compare_models(self.clients[0].discriminator, self.clients[idx].discriminator)
        print("done.")


    def _train(self, epoch):
        for idx, client in enumerate(self.clients):
            client.train_1_epoch_WGAN_GP(self.generator, self.g_optimizer)
            # if debug:
            #     print("Comparing generator")
            #     self.compare_models(temp_model, self.generator)

    def train(self):
        for epoch in range(n_epochs):
            # asyncio.run(self._async_train(epoch))
            self._train(epoch)
            if share_D:
                self.average()
            

    def val(self):
        server.clients[0].val()

            
    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0 and debug:
            print('Models match perfectly! :)')
        
class Client():
    # each client has one generator and discriminator
    def __init__(self, cid):
        self.discriminator = Discriminator(3)
        self.batches_done = 0
        self.id = cid
        self.cuda = cuda
        self.batch_size = batch_size
        self.lambda_term = 10
        self.g_iter = 1
        # Optimizers
        self.b1 = 0.5
        self.b2 = 0.999
        self.learning_rate = 1e-4
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        
        self.data = get_infinite_batches(trainloader[self.id])

        if cuda:
            self.discriminator.cuda()

        if load_D:
            self.load_model("discriminator_pretrain.pkl")
        # init own dataset based on cid

    def load_model(self, D_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        self.discriminator.load_state_dict(torch.load(D_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)



    def save_model(self, generator, discriminator):
        torch.save(generator.state_dict(), './generator_pid_{}.pkl'.format(self.id))
        torch.save(discriminator.state_dict(), './discriminator_pid_{}.pkl'.format(self.id))
        print('Models save to ./generator_pid_{}.pkl & ./discriminator_pid_{}.pkl '.format(self.id, self.id))

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def train_1_epoch_WGAN_GP(self, generator: Generator, g_optimizer):
        self.t_begin = t.time()

        

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            self.cuda_index=0
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

    # for g_iter in range(1):
        # Requires grad, Generator requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = True

        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        for d_iter in range(5):
            self.discriminator.zero_grad()

            images = self.data.__next__()
            # Check for batch to have full batch_size
            if (images.size()[0] != self.batch_size):
                continue

            z = torch.rand((self.batch_size, 100, 1, 1))

            images, z = self.get_torch_variable(images), self.get_torch_variable(z)

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            d_loss_real = self.discriminator(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)

            # Train with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

            fake_images = generator(z)
            d_loss_fake = self.discriminator(fake_images)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
            gradient_penalty.backward()


            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            self.d_optimizer.step()
            if debug:
                print(f'  Discriminator iteration: {d_iter+1}/{5}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

        # Generator update
        for p in self.discriminator.parameters():
            p.requires_grad = False  # to avoid computation

        generator.zero_grad()
        # train generator
        # compute loss with fake images
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        fake_images = generator(z)
        g_loss = self.discriminator(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        g_cost = -g_loss
        g_optimizer.step()
        print(f'Generator iteration: {self.g_iter}/{n_epochs}, g_loss: {g_loss}')
        # Saving model and sampling images every 1000th generator iterations
        if (self.g_iter) % 100 == 0:
            # self.save_model(generator, self.discriminator)
            # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
            # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
            # # This way Inception score is more correct since there are different generated examples from every class of Inception model
            # sample_list = []
            # for i in range(125):
            #     samples  = self.data.__next__()
            # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
            # #     samples = self.G(z)
            #     sample_list.append(samples.data.cpu().numpy())
            # #
            # # # Flattening list of list into one list
            # new_sample_list = list(chain.from_iterable(sample_list))
            # print("Calculating Inception Score over 8k generated images")
            # # # Feeding list of numpy arrays
            # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
            #                                       resize=True, splits=10)

            if not os.path.exists('training_result_images/'):
                os.makedirs('training_result_images/')

            # Denormalize images and save them in grid 8x8
            z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
            samples = generator(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = utils.make_grid(samples)
            utils.save_image(grid, 'training_result_images/img_generatori_iter_{}_pid_{}.png'.format(str(self.g_iter).zfill(3), self.id))

            # Testing
            time = t.time() - self.t_begin
            #print("Real Inception score: {}".format(inception_score))
            print("Generator iter: {}".format(self.g_iter))
            print("Time {}".format(time))
        
                # Write to file inception_score, gen_iters, time
                #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                #self.file.write(output)




        self.g_iter += 1
        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model(generator, self.discriminator)
        return generator

        
        

# make N clients
client_list = [Client(x) for x in range(client_cnt)]

# create server
server = Server(client_list)


# start training
server.train()
# server.val()