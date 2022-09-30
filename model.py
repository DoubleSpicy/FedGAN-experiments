import sys, os
sys.path.append(os.getcwd())

# from image_processor import CustomColorChange


import argparse
# import enum
from typing import OrderedDict
import numpy as np
# import math
# import random

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
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil

import matplotlib.pyplot as plt
os.makedirs("../../data/mnist", exist_ok=True)

dir = './dataset_transform'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

import cv2 as cv
from PIL import Image

def BGR2PIL(img):
    # BGR opencv -> PIL
    cv.cvtColor(img, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


class CustomColorChange:
    ''' Change background color
        https://pytorch.org/vision/stable/transforms.html'''
    def __init__(self, debug=False):
        self.debug = debug
        return
    
    def __call__(self, imgPIL):
        imgRGB = np.array(imgPIL)
        imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)
        # imgRGB = cv.cvtColor(imgRGB, cv.COLOR_BGR2BGR)
        imgHSV = cv.cvtColor(imgBGR, cv.COLOR_RGB2HSV)
        if self.debug:
            cv.imwrite('testimgHSV.jpg',cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR))

        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 125])
        mask = cv.inRange(imgHSV, lower, upper)
        cv.imwrite('testimgMask.jpg', cv.cvtColor(mask, cv.COLOR_GRAY2BGR))

        color_mask = np.zeros((28, 28, 3), np.uint8)
        color_mask[:] = (125, 180, 255) # red
        if self.debug:
            cv.imwrite('testimgColorMask.jpg', cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        color_mask = cv.bitwise_and(color_mask, color_mask, mask=mask)
        imgBGR = cv.add(imgBGR, cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        if self.debug:
            cv.imwrite('testimgMasked.jpg', cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB))
        pil_img = BGR2PIL(imgBGR)
        if self.debug:
            pil_img.save('testimgPIL.jpg')
        return pil_img


# dataset = datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.Grayscale(3), CustomColorChange(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#     )
# # transforms.ToTensor(),  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
# i = 0

# def save_image(numpy_array, name):
#     # save this numpy_array image as jpg
#     numpy_array = numpy_array.astype(np.uint8)
#     img = Image.fromarray(numpy_array)
#     img.save(name + ".jpg")

def nparray_to_PIL(nparr):
    # convert for opencv use
    nparr *= 255
    return Image.fromarray(nparr.astype(np.uint8))


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
n_epochs = 100
batch_size = 8
lr = 0.00005
n_cpu = 8
latent_dim = 100
img_size = 28
channels = 3
n_critic = 5
clip_value = 0.01
sample_interval = 400
client_cnt = 8
share_D = True

dataset = datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]),
    )
print(len(dataset))
# select digit = 1 and 9 only
idx = (dataset.targets==1) | (dataset.targets==9) 
dataset.targets = dataset.targets[idx]
dataset.data = dataset.data[idx]
print(len(dataset))

# split
test_len = int(len(dataset)*0.8)
trainset, testset = random_split(dataset, [test_len, len(dataset)-test_len], generator=torch.Generator().manual_seed(42))
print("train and test set size: ", len(trainset), len(testset))

# split n sets for the train and test set
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

trainset = random_split(trainset, cal_split_lengths(len(trainset), client_cnt), generator=torch.Generator().manual_seed(42))
testset = random_split(testset, cal_split_lengths(len(testset), client_cnt), generator=torch.Generator().manual_seed(42))
visualize_length(trainset)
visualize_length(testset)

# apply color filter on each trainset
for i in range(len(trainset)):
    trainset[i].dataset.transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(3),
        CustomColorChange(), 
        # transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

trainloader = [torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
                                            shuffle=True, num_workers=2) for i in range(client_cnt)]
testloader = [torch.utils.data.DataLoader(testset[i], batch_size=batch_size,
                                            shuffle=True, num_workers=2) for i in range(client_cnt)]

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class Server():
    # for doing model averaging after updating local ones
    def __init__(self, client_list:list()):
        self.clients = client_list

    def recv(self):
        pass

    def average(self):
        # do we need to accelerate this for large model/client counts with extern C++ calls?
        print("averaging models on %d clients..."%(len(self.clients)))
        averaged_gen = OrderedDict()
        averaged_dis = OrderedDict()
        for it, client in enumerate(self.clients):
            gen_params = client.generator.state_dict()
            dis_params = client.discriminator.state_dict()
            coefficient = 1/len(self.clients)
            for idx, val in gen_params.items():
                if idx not in averaged_gen:
                    averaged_gen[idx] = 0
                averaged_gen[idx] += val * coefficient
            if share_D:
                for idx, val in dis_params.items():
                    if idx not in averaged_dis:
                        averaged_dis[idx] = 0
                    averaged_dis[idx] += val * coefficient

        print("updating models for all clients")
        for idx, client in enumerate(self.clients):
            # client.generator.model = nn.Linear(1,1)
            self.clients[idx].generator.load_state_dict(gen_params)
            # client.discriminator.model = nn.Linear(1,1)
            if share_D:
                self.clients[idx].discriminator.load_state_dict(dis_params)
        print("done.")

    async def _async_train(self, epoch):
        tasks = [client.train_1_epoch(epoch+1) for idx, client in enumerate(self.clients)]
        await asyncio.wait(tasks)
        # for idx, client in enumerate(self.clients):
        #     client.train_1_epoch(epoch+1)
    def _train(self, epoch):
        for idx, client in enumerate(self.clients):
            client.train_1_epoch(epoch+1)

    def train(self):
        for epoch in range(n_epochs):
            # asyncio.run(self._async_train(epoch))
            self._train(epoch)
            self.average()

    def val(self):
        server.clients[0].val()
            
        
class Client():
    # each client has one generator and discriminator
    def __init__(self, cid):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.batches_done = 0
        self.id = cid

        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # init own dataset based on cid

        

    def train_1_epoch(self, epoch):
        # attrs = vars(self)
        for i, (imgs, _) in enumerate(trainloader[self.id]):

            # Configure input
            
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            fake_imgs = self.generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

            loss_D.backward()
            self.optimizer_D.step()

            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)


            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = self.generator(z)
                # Adversarial loss
                loss_G = -torch.mean(self.discriminator(gen_imgs))

                loss_G.backward()
                self.optimizer_G.step()

                print(
                    "[cid: %d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (self.id, epoch, n_epochs, self.batches_done % len(trainloader[self.id]), len(trainloader[self.id]), loss_D.item(), loss_G.item())
                )

            if self.batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/pid%d_batch%d.png" % (self.id, self.batches_done), nrow=5, normalize=True)
            

            self.batches_done += 1

    def val(self):
        loss_list = list()
        for i, (imgs, _) in enumerate(testloader[self.id]):
            # validate the generator
            self.optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            # Generate a batch of images
            gen_imgs = self.generator(z)
            # Adversarial loss
            loss_G = -torch.mean(self.discriminator(gen_imgs))
            loss_list.append(loss_G.item())
        print("validation:", "[G loss: %f]"%(sum(loss_list)/len(loss_list)))
        
        

# make N clients
client_list = [Client(x) for x in range(client_cnt)]

# create server
server = Server(client_list)


# start training
server.train()
server.val()