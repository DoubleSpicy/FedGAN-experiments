"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.GAN import Generator, Discriminator, train_1_epoch_D, train_1_epoch_G, send_params, recv_params
from utils.loader import load_dataset, get_infinite_batches, load_model

import argparse

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

size = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAN', required=False)
parser.add_argument('--dataset', type=str, default='MNIST', required=False)
parser.add_argument('--n_epochs', type=int, default=10000, required=False)
parser.add_argument('--batch_size', type=int, default=64, required=False)
parser.add_argument('--channels', type=int, default=3, required=False)
parser.add_argument('--n_critic', type=int, default=5, required=False)
parser.add_argument('--client_cnt', type=int, default=torch.cuda.device_count(), required=False)
parser.add_argument('--share_D', type=str, default='False', required=False)
parser.add_argument('--load_G', type=str, default='False', required=False)
parser.add_argument('--load_D', type=str, default='False', required=False)
parser.add_argument('--debug', type=str, default='True', required=False)
parser.add_argument('--proportion', type=float, default=2, required=False)
parser.add_argument('--random_colors', type=str, default='1_per_group', required=False)
parser.add_argument('--resize_to', type=int, default=32, required=False)
parser.add_argument('--time_now', type=str, default='', required=False)
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
proportion = args.proportion
random_colors = args.random_colors
resize_to = 32

os.makedirs("runs", exist_ok=True)
root = "runs/" + ''
args_dict = dict(vars(args))
for i, ii in args_dict.items():
    print(i, ii)
#     root += (i + '_' + str(ii) + '_')
root += ('_' + model +'_' + str(proportion) + '_' + str(share_D) + '_' + dataset_name)
# print(share_D)
os.makedirs(root, exist_ok=True)



    
def run(rank, size):
    group = 'a' if rank < proportion else 'b'
    trainloader, img_shape = load_dataset(dataset_name=dataset_name,
                    random_colors='1_per_group', 
                    client_cnt=1, 
                    channels=channels, 
                    batch_size=batch_size,
                    colors = None, 
                    debug=debug,
                    group=group,
                    root='.')
    print(f'rank: {dist.get_rank()}, dataloader group: {group}')
    print('device:', rank%size)
    model_G = Generator(img_shape=img_shape).to(rank % size)
    model_D = Discriminator(img_shape=img_shape).to(rank % size)
    print('share_D:',share_D)

    for i in range(1, n_epochs+1):
        print(f'iter {i}')
        train_1_epoch_D(model_G, model_D, cuda=True, n_critic=n_critic, data=get_infinite_batches(trainloader[0]),
        batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size)
        for j in range(size):
            rank_list = [r for r in range(0, size)]
            send, recv = j, rank_list[(j+1) % len(rank_list)]
            if rank == recv:
                recv_buf = torch.tensor([rank])
                dist.recv(recv_buf, src=send) # BLOCKING TO WAIT FINISH G TRAINING
                load_model(filename=f'{root}/generator_pid_{send}.pkl', model=model_G)
                print(f'{recv}: generator loaded from {root}/generator_pid_{send}.pkl')
            elif rank == send:
        #         train_1_epoch_D(model_G, model_D, cuda=True, n_critic=n_critic, data=get_infinite_batches(trainloader[0]),
        # batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size)
                train_1_epoch_G(model_G, model_D, cuda=True, n_critic=n_critic, 
                data=get_infinite_batches(trainloader[0]), batch_size=batch_size, 
                debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, 
                id=send, root=root,size=size)
                send_buf = torch.tensor([rank]) # OK MESSAGE
                dist.send(send_buf, dst=recv)
            dist.barrier()
        if share_D:
            average_params(model_D)

def average_params(model: torch.nn.Module):
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, async_op=False)
        param.data /= size

if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()