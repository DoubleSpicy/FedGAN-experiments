#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 2 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_2to2_CelebA.txt
