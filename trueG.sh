#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --share_D True --model GAN > ./logs/cifarTrueGAN.txt