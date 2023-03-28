#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --share_D False --model GAN > ./logs/cifarFalse.txt