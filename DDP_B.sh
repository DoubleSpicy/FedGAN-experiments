#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 1 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 2 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_2to2_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 3 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_3to1_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 0 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_0to4_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 4 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_4to0_CelebA.txt