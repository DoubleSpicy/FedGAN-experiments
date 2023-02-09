"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.loader import load_dataset, get_infinite_batches, load_model, save_model

import argparse



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

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
    from models.WGAN_GP import Generator, Discriminator, update, save_sample
os.makedirs("runs", exist_ok=True)
root = "runs/" + ''
args_dict = dict(vars(args))
for i, ii in args_dict.items():
    print(i, ii)
#     root += (i + '_' + str(ii) + '_')
root += ('_' + model +'_' + str(proportion) + '_' + str(share_D) + '_' + dataset_name + '_AvgMod_' + str(avg_mod) + '_delay_' + str(delay))
# print(share_D)
os.makedirs(root, exist_ok=True)

if average_method == 'wasserstein':
    from barycenters import wasserstein

negative_proportion = [0.5, 0.1, 0.1, 0.1]

def run(rank, size):
    group = 'a' if rank < proportion else 'b'
    if random_colors == 'all_random':
        group = 'a' if (rank+1)/size <= proportion else 'b'
    print(f'rank: {rank} | group: {group}')
    trainloader, img_shape = load_dataset(dataset_name=dataset_name,
                    random_colors=random_colors, 
                    client_cnt=1, 
                    channels=channels, 
                    batch_size=batch_size,
                    colors = None, 
                    debug=debug,
                    root='.',
                    P_Negative=negative_proportion[rank],
                    rank = dist.get_rank())
    print(f'rank: {dist.get_rank()}, dataloader ratio: {str(1-negative_proportion[rank])}:{negative_proportion[rank]}')
    print('device:', rank%size)
    model_G = Generator(img_shape=img_shape).to(rank % size)
    if load_G:
        load_model(f'{root}/generator_pid_{dist.get_rank()}.pkl', model_G)
    model_D = Discriminator(img_shape=img_shape).to(rank % size)
    if load_D:
        load_model(f'{root}/discriminator_pid_{dist.get_rank()}.pkl', model_D)
    print('share_D:',share_D)

    if not eval_only:
        for i in range(1, n_epochs+1):
            print(f'iter {i}')
            update(model_G, model_D, cuda=True, n_critic=int(n_critic/2), data=get_infinite_batches(trainloader[0]),
            batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size, D_only=True)
            # average_params(model_G, 'G')
            # if share_D:
            #     average_params(model_D ,'D')
            update(model_G, model_D, cuda=True, n_critic=int(n_critic/2), data=get_infinite_batches(trainloader[0]),
            batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size, D_only=False)
            if i % avg_mod == 0:
                average_params(model_G, 'G')
            if share_D:
                average_params(model_D, 'D')
            if dist.get_rank() == 0 and (i % 100 == 0 or i == n_epochs):
                save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=i)
            save_model(generator=model_G, discriminator=model_D, id=rank, root=root, averaged=True)
            
    if eval_only:
        average_params(model_G, 'G')
        if dist.get_rank() == 0:
            save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=n_epochs)


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
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, avg_only))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
