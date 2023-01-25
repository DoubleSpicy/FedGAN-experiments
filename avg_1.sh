#!/bin/bash
### THIS IS THE TASK FOR CHECKING DIFFERENT AVERAGING FREQ
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D True --proportion 1 --avg_mod 1 --delay D > ./logs/DDP_WGAN-GP_4GPU_YESshare_delayMod1_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D True --proportion 1 --avg_mod 4 --delay D > ./logs/DDP_WGAN-GP_4GPU_YESshare_delayMod4_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D True --proportion 1 --avg_mod 16 --delay D > ./logs/DDP_WGAN-GP_4GPU_YESshare_delayMod16_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D True --proportion 1 --avg_mod 64 --delay D > ./logs/DDP_WGAN-GP_4GPU_YESshare_delayMod64_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D True --proportion 1 --avg_mod 256 --delay D > ./logs/DDP_WGAN-GP_4GPU_YESshare_delayMod256_D_eyeglass_1to3_CelebA.txt

python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D False --proportion 1 --avg_mod 1 --delay D > ./logs/DDP_WGAN-GP_4GPU_NOshare_delayMod1_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D False --proportion 1 --avg_mod 4 --delay D > ./logs/DDP_WGAN-GP_4GPU_NOshare_delayMod4_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D False --proportion 1 --avg_mod 16 --delay D > ./logs/DDP_WGAN-GP_4GPU_NOshare_delayMod16_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D False --proportion 1 --avg_mod 64 --delay D > ./logs/DDP_WGAN-GP_4GPU_NOshare_delayMod64_D_eyeglass_1to3_CelebA.txt
python3 DDPtraining.py --n_epochs 2048 --model WGAN-GP --share_D False --proportion 1 --avg_mod 256 --delay D > ./logs/DDP_WGAN-GP_4GPU_NOshare_delayMod256_D_eyeglass_1to3_CelebA.txt