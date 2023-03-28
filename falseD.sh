#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --share_D False --model DCGAN > ./logs/cifarFalse.txt