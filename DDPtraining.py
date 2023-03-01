"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.loader import load_dataset, get_infinite_batches, load_model, save_model
from utils.lossLogger import lossLogger, FIDLogger
import argparse
import shutil
import numpy as np

def init_process(rank, size, trainloader, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, trainloader)

size = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='WGAN-GP', required=False)
parser.add_argument('--dataset', type=str, default='CelebA', required=False)
parser.add_argument('--n_epochs', type=int, default=10000, required=False)
parser.add_argument('--batch_size', type=int, default=64, required=False)
parser.add_argument('--channels', type=int, default=3, required=False)
parser.add_argument('--n_critic', type=int, default=10, required=False)
parser.add_argument('--client_cnt', type=int, default=torch.cuda.device_count(), required=False)
parser.add_argument('--share_D', type=str, default='True', required=False)
parser.add_argument('--load_G', type=str, default='False', required=False)
parser.add_argument('--load_D', type=str, default='False', required=False)
parser.add_argument('--eval_only', type=str, default='False', required=False)
parser.add_argument('--debug', type=str, default='True', required=False)
parser.add_argument('--proportion', type=float, default=0.4, required=False)
parser.add_argument('--random_colors', type=str, default='all_random', required=False)
parser.add_argument('--resize_to', type=int, default=32, required=False)
parser.add_argument('--time_now', type=str, default='', required=False)
parser.add_argument('--average_method', type=str, default='euclidean', required=False)
parser.add_argument('--avg_mod', type=int, default=1, required=False)
parser.add_argument('--delay', type=str, default='', required=False)
args = parser.parse_args()
n_epochs = args.n_epochs
dataset_name = args.dataset
batch_size = args.batch_size
model = args.model
channels = args.channels
n_critic = args.n_critic
client_cnt = args.client_cnt
share_D = True if args.share_D == 'True' else False
debug = True if args.debug == 'True' else False
load_G = True if args.load_G == 'True' else False
load_D  = True if args.load_D == 'True' else False
eval_only = True if args.eval_only == 'True' else False
proportion = args.proportion
random_colors = args.random_colors
average_method = args.average_method
avg_mod = args.avg_mod
delay = args.delay
resize_to = 32

assert average_method in ['euclidean', 'wasserstein']
assert model in ['GAN', 'WGAN-GP']
if model == 'GAN':
    from models.GAN import Generator, Discriminator, update, update_G, send_params, recv_params, save_sample
if model == 'WGAN-GP':
    from models.WGAN_GP import Generator, Discriminator, update, save_sample, generate_images, calculate_FID
os.makedirs("runs", exist_ok=True)
root = "runs/" + ''
args_dict = dict(vars(args))
# for i, ii in args_dict.items():
#     print(i, ii)
#     root += (i + '_' + str(ii) + '_')
root += ('_' + model +'_' + str(proportion) + '_' + str(share_D) + '_' + dataset_name + '_AvgMod_' + str(avg_mod) + '_delay_' + str(delay))
# print(share_D)
# if os.path.exists(root):
#     shutil.rmtree(root)
# os.makedirs(root, exist_ok=True)

from utils.fid.fid_score import compute_FID

def run(rank, size, trainloader):
    group = 'a' if rank < proportion else 'b'
    if random_colors == 'all_random':
        group = 'a' if (rank+1)/size <= proportion else 'b'
    print(f'rank: {rank} | group: {group}')
    img_shape = [3, 64, 64]
    # if not eval_only:
    print('device:', rank%size)

    # loggers
    loss_logger = lossLogger(root, rank, ['d_loss_fake', 'd_loss_real', 'g_loss'], 'iterations', 'loss')
    fid_cols = [f'vs_data{i}' for i in range(size)]
    fid_cols.append('vs_dataAll')
    FID_logger = FIDLogger(dir=root, id=rank, x_label='iterations(*100)', y_label='FID', columns=fid_cols)
    print(FID_logger.columns)
    # loss_logger_averaged = lossLogger(root, rank, ['d_loss_fake', 'd_loss_real', 'g_loss'], 'iterations(*100)', 'averaged G loss')
    FID_logger_averaged = FIDLogger(dir=root, id=777+rank, x_label='iterations(*100)', y_label='FID score', columns=fid_cols)
    model_G = Generator(img_shape=img_shape).to(rank % size)
    if load_G:
        load_model(f'{root}/generator_pid_{dist.get_rank()}.pkl', model_G)
    model_D = Discriminator(img_shape=img_shape).to(rank % size)
    if load_D:
        load_model(f'{root}/discriminator_pid_{dist.get_rank()}.pkl', model_D)
    print('share_D:',share_D)

    if not eval_only:
        precompute_npz(rank=rank, trainloader=trainloader)
        for i in range(1, n_epochs+1):
            print(f'iter {i}')
            update(model_G, model_D, cuda=True, n_critic=int(n_critic/2), data=get_infinite_batches(trainloader),
            batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size, D_only=True, loss_logger=loss_logger)
            # average_params(model_G, 'G')
            # if share_D:
            #     average_params(model_D ,'D')
            update(model_G, model_D, cuda=True, n_critic=int(n_critic/2), data=get_infinite_batches(trainloader),
            batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size, D_only=False, loss_logger=loss_logger)
            if i % 100 == 0 or i == 1:
                FID_logger.concat(calculate_FID(root=root, generator=model_G, discriminator=model_D, device=rank, rank=rank))
            if i % avg_mod == 0:
                average_params(model_G, 'G')
            if share_D:
                average_params(model_D, 'D')
            if i % 100 == 0 or i == n_epochs: # handle averaged stuff
                if dist.get_rank() == 0:
                    save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=i)
                print(f'rank {rank} computing FID after averaged')
            if i % 100 == 0 or i == 1:
                score = calculate_FID(root=root, generator=model_G, discriminator=model_D, device=rank, rank=rank)
                FID_logger_averaged.concat(score)
            save_model(generator=model_G, discriminator=model_D, id=rank, root=root, averaged=True)
            
            # print(f'{rank} vs dataAll: {score}')
    if eval_only:
        # average_params(model_G, 'G')
        if dist.get_rank() == 0:
            save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=n_epochs)

def precompute_npz(rank: int, trainloader):
    if os.path.exists(os.path.join(root, f'data{rank}.npz')):
        return
    path = os.path.join(root, f'data{rank}')
    allPath = os.path.join(root, f'dataAll')
    if os.path.exists(path):
        shutil.rmtree(path)
    if rank == 0 and os.path.exists(allPath):
        shutil.rmtree(allPath)
    os.makedirs(path, exist_ok=True)
    os.makedirs(allPath, exist_ok=True)
    batch = 0
    dist.barrier()
    for _, (image, _) in enumerate(trainloader):
        image = torch.unbind(image)
        for j in range(len(image)):
            torchvision.utils.save_image(image[j], path + "/" + str(batch*64+j).zfill(6) + '.png')
            torchvision.utils.save_image(image[j], allPath + "/" + str(batch*64+j).zfill(6) + 'device' + str(rank) + '.png')
        batch += 1
    compute_FID([path, os.path.join(root, f'data{rank}.npz')], save_stat=True, rank=rank)
    if rank == 0:
        compute_FID([allPath, os.path.join(root, f'dataAll.npz')], save_stat=True, rank=rank)
    dist.barrier()
        
def average_params(model: torch.nn.Module, str):
    print(f'rank:{dist.get_rank()}| averaging {str}')
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, async_op=False)
        param.data /= size

def avg_only(rank, size):
    print('*****one off averaging...*****')
    model_G = Generator([3, 64, 64])
    model_D = Discriminator([3, 64, 64])
    load_model(f'{root}/generator_pid_{dist.get_rank()}.pkl', model_G)
    load_model(f'{root}/discriminator_pid_{dist.get_rank()}.pkl', model_D)
    average_params(model_G, 'G')
    if share_D:
        average_params(model_D, 'D')
    if rank == 0:
        save_model(generator=model_G, discriminator=model_D, id=rank, root=root, averaged=True)

if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")
    trainloader = []
    if not eval_only:
        ratios = np.random.default_rng().dirichlet((10, 10, 10, 10, 10, 10, 10, 10, 10, 10), size)
        print(ratios)
        trainloader = load_dataset(root=root,
                                    dataset='CIFAR10',
                                    client_ratio = ratios)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, trainloader[rank], run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
