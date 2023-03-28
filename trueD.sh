#!/bin/bash
mkdir -p ./logs
python3 DDPtraining.py --share_D True --model DCGAN > ./logs/cifarTrue.txt