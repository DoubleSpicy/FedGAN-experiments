python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D False --proportion 4 > ./logs/DDP_GAN_4GPU_NOTshare_D_eyeglass_4to0_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model GAN --share_D True --proportion 4 > ./logs/DDP_GAN_4GPU_YESshare_D_eyeglass_4to0_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D False --proportion 4 > ./logs/DDP_WGAN-GP_4GPU_NOTshare_D_eyeglass_4to0_CelebA.txt
python3 DDPtraining.py --n_epochs 3000 --model WGAN-GP --share_D True --proportion 4 > ./logs/DDP_WGAN-GP_4GPU_YESshare_D_eyeglass_4to0_CelebA.txt