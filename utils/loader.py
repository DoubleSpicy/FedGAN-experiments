
import torchvision.transforms as transforms
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

from utils.datasets import celeba, TinyImageNet, equalize

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



def save_model(generator, discriminator, id, root):
    model_path = os.path.join(os.getcwd(), root)
    torch.save(generator.state_dict(), '{}/generator_pid_{}.pkl'.format(model_path, id))
    torch.save(discriminator.state_dict(), '{}/discriminator_pid_{}.pkl'.format(model_path, id))
    print('Models save to {}/generator_pid_{}.pkl & {}/discriminator_pid_{}.pkl '.format(model_path, id, model_path, id))

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
                group='a',
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
            if group == 'a':
                dataset = celeba(root_dir='../data/', 
                                attr_data='list_attr_celeba.txt', 
                                img_path='img_align_celeba', 
                                attr_filter=['+Eyeglasses'],
                                transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5, ), (0.5, ))])
                                ,proportion=P_Negative)
            elif group == 'b':
                dataset = celeba(root_dir='../data/', 
                                attr_data='list_attr_celeba.txt', 
                                img_path='img_align_celeba', 
                                attr_filter=['+Eyeglasses'],
                                transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5, ), (0.5, ))])
                                ,proportion=1-P_Negative)
            img_shape = [3, 64, 64]
            trainset.append(dataset)

        # print(len(dataset))
        
    elif random_colors == '1_per_group':
        trainset = []
        for i in range(client_cnt):
            assert dataset_name in ['MNIST', 'CelebA', 'TinyImageNet']
            if dataset_name == 'MNIST':
                dataset = datasets.MNIST("../data/mnist",
                                        train=True,
                                        download=True,
                                        transform = transforms.Compose([
                                        transforms.Resize(64),
                                        transforms.Grayscale(channels),
                                        # CustomColorChange(colors=colors, all_random=False, debug=debug),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))
                                        ]))
                if group == 'a':
                    indices = dataset.targets == 0 # if you want to keep images with the label 5
                else:
                    indices = dataset.targets == 1 # if you want to keep images with the label 5
                dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
                img_shape = [3, 64, 64]
            elif dataset_name == 'CelebA':
                if (group == 'a'):
                    # print("a", i)
                    dataset = celeba(root_dir='../data/', 
                                    attr_data='list_attr_celeba.txt', 
                                    img_path='img_align_celeba', 
                                    attr_filter=['+Eyeglasses'],
                                    transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))])
                                    ,proportion=0)
                else:
                    # print("b", i)
                    dataset = celeba(root_dir='../data/', 
                                    attr_data='list_attr_celeba.txt', 
                                    img_path='img_align_celeba', 
                                    attr_filter=['-Eyeglasses'],
                                    transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, ), (0.5, ))])
                                    ,proportion=0)
                    if i > 0:
                        equalize(trainset[0], dataset)
            elif dataset_name == 'TinyImageNet':
                if (group == 'a'):
                    dataset = TinyImageNet(root_dir='../data/tiny-imagenet-200/', 
                            attr_data='classes.csv', 
                            img_path='train',
                            attr_filter=['+ox'],
                            transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, ), (0.5, ))]))
                    
                else:
                    dataset = TinyImageNet(root_dir='../data/tiny-imagenet-200/', 
                            attr_data='classes.csv', 
                            img_path='train',
                            attr_filter=['+mushroom'],
                            transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, ), (0.5, ))]))
                # print(dataset.attribute_data)
            img_shape = [3, 64, 64]
            # print(len(dataset))
            trainset.append(dataset)
    # print('=======================')
    trainloader = list()
    for i in range(0, client_cnt):
        # trainset[i]
        trainloader.append(torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
                                                shuffle=True, drop_last=True))
    return trainloader, img_shape