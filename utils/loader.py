
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

def load_model(filename, model):
    model_path = os.path.join(os.getcwd(), filename)
    model.load_state_dict(torch.load(model_path))
    print('model loaded from {}-'.format(model_path))



def save_model(generator, discriminator, id, root):
    torch.save(generator.state_dict(), '{}/generator_pid_{}.pkl'.format(root, id))
    torch.save(discriminator.state_dict(), '{}/discriminator_pid_{}.pkl'.format(root, id))
    print('Models save to {}/generator_pid_{}.pkl & {}/discriminator_pid_{}.pkl '.format(root, id, root, id))

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
        imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)

        # imgRGB = cv.cvtColor(imgRGB, cv.COLOR_BGR2BGR)
        imgHSV = cv.cvtColor(imgBGR, cv.COLOR_RGB2HSV)
        if self.debug:
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
        if self.debug:
            cv.imwrite('testimgColorMask.jpg', cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        color_mask = cv.bitwise_and(color_mask, color_mask, mask=mask)
        imgBGR = cv.add(imgBGR, cv.cvtColor(color_mask, cv.COLOR_HSV2BGR))
        if self.debug:
            cv.imwrite('testimgMasked.jpg', cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB))
        pil_img = BGR2PIL(imgBGR)
        if self.debug:
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




def prob_choose(target, digits: list, prob: list):
    for i in range(len(digits)):
        if target == digits[i]:
            return prob_true(0.9) and random.uniform(0, 1) <= prob[i]
    return False

def load_dataset(random_colors='1_per_group', 
                client_cnt=5, 
                channels=3, 
                batch_size=64,
                colors:tuple = None, 
                debug=False,
                proportion=0.2,
                root=''):
    if random_colors == 'all_random':
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

        print(len(dataset))
        trainset = random_split(dataset, cal_split_lengths(len(dataset), client_cnt), generator=torch.Generator().manual_seed(42))
    elif random_colors == '1_per_group':
        trainset = []
        for i in range(client_cnt):
            trainset.append(
                datasets.MNIST(
                "{}/data/mnist".format(root),
                train=True,
                download=True,
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.Grayscale(channels),
                    CustomColorChange(colors=colors, all_random=False, debug=debug),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, )),
                ])
            ))

        for i in range(math.floor(client_cnt*(1-proportion))):
            print("a")
            idx = [prob_choose(i, [0], [1]) for i in trainset[i].targets] # random sampling
            trainset[i].targets = trainset[i].targets[idx]
            trainset[i].data = trainset[i].data[idx]
            print("trainset[i] length:", len(trainset[i]))

        for i in range(math.floor(client_cnt*(1-proportion)), client_cnt):
            print("b")
            idx = [prob_choose(i, [1], [1]) for i in trainset[i].targets] # random sampling
            trainset[i].targets = trainset[i].targets[idx]
            trainset[i].data = trainset[i].data[idx]
            print("trainset[i] length:", len(trainset[i]))

    print('=======================')
    trainloader = list()
    for i in range(0, client_cnt):
        trainloader.append(torch.utils.data.DataLoader(trainset[i], batch_size=batch_size,
                                                shuffle=True, drop_last=True))
    return trainloader