#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --share_D False --model WGAN-GP > ./logs/cifarFalse.txt