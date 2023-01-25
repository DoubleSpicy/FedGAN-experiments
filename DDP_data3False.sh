#!/bin/bash
# tiny image net again!!!
mkdir -p ./logs
python3 DDPtraining.py --dataset TinyImageNet --n_epochs 3000 --model WGAN-GP --share_D False --proportion 2.0 > ./logs/DDP_GAN_4GPU_YESshare_D_OxMushroom_2to2_TinyImageNet.txt
python3 DDPtraining.py --dataset TinyImageNet --n_epochs 3000 --model WGAN-GP --share_D False --proportion 1.0 > ./logs/DDP_GAN_4GPU_YESshare_D_OxMushroom_2to2_TinyImageNet.txt
