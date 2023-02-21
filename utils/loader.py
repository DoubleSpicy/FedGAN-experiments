
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, utils
from torch.autograd import Variable
from torch import autograd
import torch
import random
import numpy as np
import cv2 as cv
from PIL import Image
import math

import os

from utils.datasets import celeba, TinyImageNet, equalize, CelebA, splitCelebA, initCIFAR10_dirichlet

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _ )in enumerate(data_loader):
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

def load_model(filename, model):
    model_path = os.path.join(os.getcwd(), filename)
    model.load_state_dict(torch.load(model_path))
    print('model loaded from {}-'.format(model_path))



def save_model(generator, discriminator, id, root, averaged=False):
    model_path = os.path.join(os.getcwd(), root)
    averagedStr = "_averaged" if averaged else ""
    torch.save(generator.state_dict(), '{}/generator{}_pid_{}.pkl'.format(model_path, averagedStr, id))
    torch.save(discriminator.state_dict(), '{}/discriminator{}_pid_{}.pkl'.format(model_path, averagedStr, id))
    print('Models save to {}/generator{}_pid_{}.pkl & {}/discriminator{}_pid_{}.pkl '.format(model_path, averagedStr, id, model_path, averagedStr, id))

class CustomColorChange():
    ''' Change background color
        https://pytorch.org/vision/stable/transforms.html'''
    def __init__(self, colors:tuple = None, all_random=False, debug=False):
        if colors == None:
            self.colors =  (random.randint(0, 255), random.randint(0, 180), random.randint(0, 255))
        else:
            self.colors = colors
        self.all_random = all_random
        self.first_time = True
        self.debug = debug
        return
    
    def __call__(self, imgPIL):
        imgRGB = np.array(imgPIL)
        imgHSV = cv.cvtColor(imgRGB, cv.COLOR_RGB2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 125])
        mask = cv.inRange(imgHSV, lower, upper)
        color_mask = np.zeros((32, 32, 3), np.uint8)
        # color_mask[:] = (125, 180, 255) # red/random
        if self.all_random:
            color_mask[:] = (random.randint(0, 255), random.randint(0, 180), random.randint(0, 255))
        else:
            color_mask[:] = self.colors # red/random
            # print(self.colors)
        color_mask = cv.bitwise_and(color_mask, color_mask, mask=mask)
        imgRGB = cv.add(imgRGB, cv.cvtColor(color_mask, cv.COLOR_HSV2RGB))
        # cv.imwrite('test.jpg', imgRGB)

        if self.first_time:
            print(self.colors)
            self.first_time = False
        return imgRGB

    



def nparray_to_PIL(nparr):
    # convert for opencv use
    nparr *= 255
    return Image.fromarray(nparr.astype(np.uint8))




def prob_choose(target, digits: list, prob: list):
    for i in range(len(digits)):
        if target == digits[i]:
            return prob_true(0.9) and random.uniform(0, 1) <= prob[i]
    return False

def load_dataset(dataset_name,
                random_colors='1_per_group', 
                client_cnt=5, 
                channels=3, 
                batch_size=64,
                colors:tuple = None, 
                debug=False,
                root='',
                P_Negative=0
                ):
    if random_colors == 'all_random':
        trainset = []
        if dataset_name == 'MNIST':
            dataset = datasets.MNIST(
                    "{}/data/mnist".format(root),
                    train=True,
                    download=True,
                    transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.Grayscale(3),
                        CustomColorChange(colors=colors, all_random=True, debug=debug),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, ), (0.5, )),
                    ]))
            trainset = random_split(dataset, cal_split_lengths(len(dataset), client_cnt), generator=torch.Generator().manual_seed(42))
            img_shape = [3, 64, 64]
        if dataset_name == 'CelebA':

            transformA = [transforms.Resize([64, 64]),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transformB =             [transforms.CenterCrop((178, 178)),
                                       transforms.Resize((64, 64)),
                                       transforms.ToTensor()]
            dataset = celeba(root_dir='../data/', 
                            attr_data='list_attr_celeba.txt', 
                            img_path='img_align_celeba', 
                            attr_filter=['+Male'],
                            transform=transforms.Compose(transformA)
                            ,proportion=P_Negative,
                            rotary=True)
            img_shape = [3, 64, 64]
            trainset.append(dataset)
    # print('=======================')
    trainloader = list()
    for i in range(0, client_cnt):
        # trainset[i]
        trainloader.append(torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
                                                shuffle=True, drop_last=True))
    return trainloader, img_shape

def load_dataset(root: str,
                dataset: str,
                client_ratio: list,
                tag_filter: list = None,
                batch_size: int = 64          
                ):
    transformA = transforms.Compose([transforms.Resize([64, 64]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'CelebA':
        size = len(client_ratio)
        datasets = [CelebA('../data/', tags=tag_filter, transform = transformA) for i in range(size)]
        splittedData = splitCelebA(datasets[0].attribute_data, client_ratio=client_ratio, tag=tag_filter)
        for i in range(size):
            datasets[i].attribute_data = splittedData[i]
        return [torch.utils.data.DataLoader(datasets[i], batch_size=batch_size, shuffle=True, drop_last=True) for i in range(size)]
    elif dataset == 'CIFAR10':
        size = torch.cuda.device_count()
        datasets = initCIFAR10_dirichlet(dirichlet_param=[10, 5, 3, 2, 3, 1, 1, 3, 4, 5]
                                        , size=size, transforms=transformA)
        return [torch.utils.data.DataLoader(datasets[i], batch_size=batch_size, shuffle=True, drop_last=True) for i in range(size)]
    return None
