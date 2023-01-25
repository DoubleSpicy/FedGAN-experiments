#!/bin/bash
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 0.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 1.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 2.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 3.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 4.0 --model GAN

python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 0.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 1.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 2.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 3.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D False --proportion 4.0 --model WGAN-GP

python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --share_D False --proportion 0.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 1.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 2.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 3.0 --model GAN
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 4.0 --model GAN

python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 0.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 1.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 2.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 3.0 --model WGAN-GP
python3 DDPtraining.py --load_D True --load_G True --eval_only True --share_D True --proportion 4.0 --model WGAN-GP
