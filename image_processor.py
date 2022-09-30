'''
this is the image processor on the dataset so that each clinet gets MNIST images of different colors
'''
import cv2 as cv

# import enum
import os
from typing import OrderedDict
import numpy as np
# import math
# import sys
# import random
from PIL import Image
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

def save_image(numpy_array, name):
    # save this numpy_array image as jpg
    numpy_array = numpy_array.astype(np.uint8)
    img = Image.fromarray(numpy_array)
    img.save(name + ".jpg")

def nparray_to_PIL(nparr):
    # convert for opencv use
    nparr *= 255
    return Image.fromarray(nparr.astype(np.uint8))


print(np.prod(28))