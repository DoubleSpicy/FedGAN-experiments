#!/bin/bash
# task for different dataset proportion
mkdir -p logs
python3 DDPtraining.py --n_epochs 1000 --model WGAN-GP --share_D True --proportion 0.2 > ./logs/DDP_WGAN-GP_4GPU_YESshare_P02_D_eyeglass_CelebA.txt
python3 DDPtraining.py --n_epochs 1000 --model WGAN-GP --share_D True --proportion 0.5 > ./logs/DDP_WGAN-GP_4GPU_YESshare_P02_D_eyeglass_CelebA.txt
python3 DDPtraining.py --n_epochs 1000 --model WGAN-GP --share_D True --proportion 0.8 > ./logs/DDP_WGAN-GP_4GPU_YESshare_P08_D_eyeglass_CelebA.txt

