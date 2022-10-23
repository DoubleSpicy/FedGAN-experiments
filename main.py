from genericpath import exists
import sys, os
sys.path.append(os.getcwd())


import argparse
# import enum
from collections import OrderedDict
import numpy as np
# import math



# import enum
import os

import numpy as np
# import math
# import sys
# import random


# import torch.nn as nn
# import torch.nn.functional as F
import torch
from utils.loader import load_dataset, get_infinite_batches, load_model

import time
global_timer = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAN', required=False)
parser.add_argument('--n_epochs', type=int, default=1000, required=False)
parser.add_argument('--batch_size', type=int, default=64, required=False)
parser.add_argument('--channels', type=int, default=3, required=False)
parser.add_argument('--n_critic', type=int, default=10, required=False)
parser.add_argument('--client_cnt', type=int, default=5, required=False)
parser.add_argument('--share_D', type=bool, default=False, required=False)
parser.add_argument('--load_G', type=bool, default=False, required=False)
parser.add_argument('--load_D', type=bool, default=False, required=False)
parser.add_argument('--debug', type=bool, default=False, required=False)
parser.add_argument('--proportion', type=float, default=0.8, required=False)
parser.add_argument('--random_colors', type=str, default='1_per_group', required=False)
parser.add_argument('--resize_to', type=int, default=32, required=False)
parser.add_argument('--time_now', type=str, default='time:N/A', required=False)
args = parser.parse_args()
print(args.debug)
n_epochs = args.n_epochs
print("n_epochs", n_epochs)
batch_size = args.batch_size
model = args.model
channels = args.channels
n_critic = args.n_critic
client_cnt = args.client_cnt
share_D = args.share_D
debug = args.debug # global debug variable
load_G, load_D = args.load_G, args.load_D
proportion = args.proportion
random_colors = args.random_colors
resize_to = 32

os.makedirs("runs", exist_ok=True)
root = "runs/" + args.time_now
os.makedirs(root, exist_ok=True)


cuda = True if torch.cuda.is_available() else False
print("is cuda available:", cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("debug:", debug)
trainloader = load_dataset(random_colors='1_per_group', 
                client_cnt=5, 
                channels=3, 
                batch_size=64,
                colors = None, 
                debug=debug,
                proportion=proportion,
                root='.')

# ====================^^^^^datasets^^^^^=========================

# model selection, load G and D architecture
if model == 'WGAN-GP':
    from models.WGAN_GP import Generator, Discriminator, train_1_epoch
    print("loaded WGAN-GP model")
elif model == 'GAN':
    from models.GAN import Generator, Discriminator, train_1_epoch
    print("loaded vanilla GAN model")



class Server():
    # for doing model averaging after updating local ones
    def __init__(self, client_list:list()):
        self.clients = client_list
        if model == 'WGAN-GP':
            self.generator = Generator(channels)
        elif model == 'GAN':
            self. generator = Generator(img_shape=(3, 32, 32))
        # return optimizer here
        self.g_iter = 1

        if cuda:
            self.generator.cuda()
        if load_G:
            load_model("generator_pretrain.pkl", self.generator)
        


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
            train_1_epoch(self.generator, 
            client.discriminator, 
            cuda=cuda, 
            n_critic=n_critic, 
            data=client.data, 
            batch_size=batch_size, 
            debug=debug, 
            n_epochs=n_epochs,
            lambda_term=client.lambda_term,
            g_iter=self.g_iter,
            id=client.id,
            root=root)
        self.g_iter += 1
    def train(self):
        for epoch in range(n_epochs):
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
        if model == 'WGAN-GP':
            self.discriminator = Discriminator(channels)
        elif model == 'GAN':
            self.discriminator = Discriminator(img_shape=(3, 32, 32))
        self.batches_done = 0
        self.id = cid
        self.cuda = cuda
        self.batch_size = batch_size
        self.lambda_term = 10
        self.g_iter = 1

        
        self.data = get_infinite_batches(trainloader[self.id])

        if cuda:
            self.discriminator.cuda()

        if load_D:
            load_model("discriminator_pretrain.pkl", self.discriminator)
        # init own dataset based on cid


# make N clients
client_list = [Client(x) for x in range(client_cnt)]

# create server
server = Server(client_list)


# start training
server.train()
# server.val()

print("total time taken:", time.time()-global_timer)
print("done")
