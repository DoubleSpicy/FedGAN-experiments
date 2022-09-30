# import asyncio

# # from time import sleep
# # from tqdm import tqdm
# # lis = ['a', 'b' ,'cdefg']
# # for a, b in tqdm(enumerate(lis)):
# #     sleep(1)
# #     print(a, b)

# async def async_print(str):
#     print(str)

# async def async_loop(limit):
#     for i in range(limit):
#         await async_print(i)

# asyncio.run(async_loop(10))
# print('done')

list1 = [1,2,3,4,5,6,7,8,9,10]
for idx, item in enumerate(list1):
    print(id(list1[idx]), id(item))
# def total_and_item(sequence):
#     total = 0
#     for i in sequence:
#         total += i
#         yield (total, i)

# list2 = list(total_and_item(list1))
# print(list2)

# def mod(sequence, id):
#     for i in range(len(sequence)):
#         if(i % id == 0):
#             yield(i)

# print(list(mod(list1, 2)))

# import random
# prob = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]
# for i in range(10):
#     print(random.choices(list1, weights=prob, k=1)[0])

# import argparse
# import enum
# import os
# from typing import OrderedDict
# import numpy as np
# import math
# import sys
# import random

# import torchvision.transforms as transforms
# from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable

# import torch.nn as nn
# import torch.nn.functional as F
# import torch

# import asyncio

# trainset = datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     )

# idx = (trainset.targets==1 and random.) | (trainset.targets==9) # select digit = 1 and 9 only
# trainset.targets = trainset.targets[idx]
# trainset.data = trainset.data[idx]

# prob = 0.1
# def rand_bool():

# print(trainset.classes)
# prob_for_classes = [prob if val == 9 else 1-prob for val in trainset.targets]
# print(prob_for_classes)

# print(trainset.targets)


# def select(trainset):
#     for i in range(len(trainset)):
#         if trainset.

# prob_for_classes = [prob if val == 9 else 1-prob for val in trainset.targets] # binary prob assignment

# prob_1 = list(random.choices(trainset.targets, weights=prob_for_classes, k=1)[0] for i in range(len(trainset)))
# print(prob_1)