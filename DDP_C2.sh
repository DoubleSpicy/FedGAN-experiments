#!/bin/bash
mkdir -p ./logs
#python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 1 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_1to3_CelebA.txt
#python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 2 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_2to2_CelebA.txt
#python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 3 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_3to1_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 0 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_0to4_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 5 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_4to0_CelebA.txt
