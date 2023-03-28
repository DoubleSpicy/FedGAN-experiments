"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

import pandas as pd

from utils.loader import load_dataset, get_infinite_batches, load_model, save_model
from utils.lossLogger import lossLogger, FIDLogger
import argparse
import shutil
import numpy as np

def init_process(rank, size, trainloader, fn, backend='gloo', port:int = 29500):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, trainloader)


size = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DCGAN', required=False)
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
assert model in ['GAN', 'WGAN-GP', 'DCGAN']
if model == 'GAN':
    from models.GAN import Generator, Discriminator, update, save_sample, calculate_FID
if model == 'WGAN-GP':
    from models.WGAN_GP import Generator, Discriminator, update, save_sample, calculate_FID
if model == 'DCGAN':
    from models.DCGAN import Generator, Discriminator, update, save_sample, weights_init, calculate_FID
    n_epochs *= n_critic
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
    if not os.path.exists('{}/training_result_images/'.format(root)):
        #print('===============\ncreated img dir=============')
        os.makedirs('{}/training_result_images/'.format(root), exist_ok=True)
    group = 'a' if rank < proportion else 'b'
    if random_colors == 'all_random':
        group = 'a' if (rank+1)/size <= proportion else 'b'
    print(f'rank: {rank} | group: {group}')
    img_shape = [3, 64, 64]
    # if not eval_only:
    print('device:', rank%size)

    # loggers
    loss_logger = lossLogger(root, rank, ['d_loss_fake', 'd_loss_real', 'g_loss'], 'iterations', 'loss')
    fid_cols = []
    for i in range(size):
        fid_cols.extend([f'fid data{i}'])
    fid_cols.extend([f'fid vs dataAll'])
    print(f'fid_cols len: {len(fid_cols)}')
    
    details_col = []
    for i in range(size):
        details_col.extend([f'fid vs data{i}', f'diff vs data{i}', f'sigma1 vs data{i}', f'sigma2 vs data{i}', f'tr_convmean vs data{i}', f'trace vs data{i}', f'frobenious norm vs data{i}'])
    details_col.extend([f'fid vs dataAll', f'diff vs dataAll', f'sigma1 vs dataAll', f'sigma2 vs dataAll', f'tr_convmean vs dataAll', f'trace vs dataAll', f'frobenious norm vs dataAll'])
    # details_averaged_col = []
    # details_averaged_col.extend([f'fid vs dataAll', f'diff vs dataAll', f'sigma1 vs dataAll', f'sigma2 vs dataAll', f'tr_convmean vs dataAll'])
    fid_details = pd.DataFrame(columns=details_col)
    fid_details_averaged = pd.DataFrame(columns=details_col)
    cross_norm_cols = []
    cross_temp = ['0', '1', '2', '3', 'All']
    for i in range(len(cross_temp)):
        for j in range(len(cross_temp)):
            cross_norm_cols.append(f'{cross_temp[i]} vs {cross_temp[j]}')
    cross_norm = pd.DataFrame(columns=cross_norm_cols)

    FID_logger = FIDLogger(dir=root, id=rank, x_label='iterations(*100)', y_label='FID', columns=fid_cols)
    print(FID_logger.columns)
    # loss_logger_averaged = lossLogger(root, rank, ['d_loss_fake', 'd_loss_real', 'g_loss'], 'iterations(*100)', 'averaged G loss')
    FID_logger_averaged = FIDLogger(dir=root, id=777+rank, x_label='iterations(*100)', y_label='FID score', columns=fid_cols)
    model_G = Generator(img_shape=img_shape).to(rank % size)
    model_D = Discriminator(img_shape=img_shape).to(rank % size)
    if model == 'DCGAN':
        model_G.apply(weights_init)
        model_D.apply(weights_init)
    if load_G:
        load_model(f'{root}/generator_pid_{dist.get_rank()}.pkl', model_G)
    if load_D:
        load_model(f'{root}/discriminator_pid_{dist.get_rank()}.pkl', model_D)
    print('share_D:',share_D)

    if not eval_only:
        
        precompute_npz(rank=rank, trainloader=trainloader)
        # if model == 'DCGAN':
        #     n_epochs *= 10
        for i in range(1, n_epochs+1):
            print(f'iter {i}')
            update(model_G, model_D, cuda=True, n_critic=10, data=get_infinite_batches(trainloader),
            batch_size=batch_size, debug=debug, n_epochs=n_epochs,lambda_term=10, g_iter=i, id=rank, root=root,size=size, D_only=False, loss_logger=loss_logger)
            print(f'rank {rank} computing FID before averaged')
            if i % 100 == 0 or i == 1:
                # fid_value, diff, tr_sigma1, tr_sigma2, tr_convmean, tr, frobenious, sigma1
                score = calculate_FID(root=root, generator=model_G, discriminator=model_D, device=rank, rank=rank, share_D=share_D)
                fid_details.loc[len(fid_details)] = [s for idx, s in enumerate(score) if idx == 0 or (idx+1) % 8 != 0]
                fid_details.to_csv(os.path.join(root, f'fid_details{rank}.csv'))
                FID_logger.concat(score[::8])
            if i % avg_mod == 0:
                average_params(model_G, 'G')
            if share_D:
                average_params(model_D, 'D')
            print('debug: i =', i)
            print(f'rank {rank} computing FID after averaged')
            if i % 100 == 0 or i == 1:
                print(f'***{rank}: computing averaged model FID***')
                score_averaged = calculate_FID(root=root, generator=model_G, discriminator=model_D, device=rank, rank=rank, share_D=share_D)
                print(f'***{rank}: done computing averaged model FID***')
                fid_details_averaged.loc[len(fid_details_averaged)] = [s for idx, s in enumerate(score_averaged) if idx == 0 or (idx+1) % 8 != 0]
                fid_details_averaged.to_csv(os.path.join(root, f'fid_details_averaged{rank}.csv'))
                FID_logger_averaged.concat(score_averaged[::8])
                gather_sigma(score[7], score_averaged[7], cross_norm, root=root, rank=rank, size=size)
            if i % 100 == 0 or i == n_epochs: # handle averaged stuff
                if dist.get_rank() == 0:
                    save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=i)
            save_model(generator=model_G, discriminator=model_D, id=rank, root=root, averaged=True)
            
            # print(f'{rank} vs dataAll: {score}')
    if eval_only:
        # average_params(model_G, 'G')
        if dist.get_rank() == 0:
            save_sample(generator=model_G, cuda_index=rank % size, root=root, g_iter=n_epochs)

def precompute_npz(rank: int, trainloader: DataLoader):
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
    count = 0
    for image in trainloader.dataset.data:
        # image = torch.unbind(image)
        # for j in range(len(image)):
        image = Image.fromarray(image)
        transformB = transforms.Compose([transforms.Resize([64, 64]),
                    transforms.ToTensor()])
        image = transformB(image)
        
    
        # image = transforms.functional.pil_to_tensor(image)
        torchvision.utils.save_image(image, path + "/" + str(count).zfill(6) + '.png')
        torchvision.utils.save_image(image, allPath + "/" + str(count).zfill(6) + 'device' + str(rank) + '.png')
        count += 1

    compute_FID([path, os.path.join(root, f'data{rank}.npz')], save_stat=True, rank=rank)
    if rank == 0:
        compute_FID([allPath, os.path.join(root, f'dataAll.npz')], save_stat=True, rank=rank)
    if os.path.exists(path):
        shutil.rmtree(path)
    dist.barrier()

def gather_sigma(local_sigma: np.ndarray, avg_sigma: np.ndarray, df: pd.DataFrame, root: str, rank: int, size: int):
    local_sigma = torch.from_numpy(local_sigma)
    avg_sigma = torch.from_numpy(avg_sigma)
    if rank == 0:
        gather_list = [local_sigma for i in range(size)]
        dist.gather(local_sigma, gather_list=gather_list)
        gather_list.append(avg_sigma)
        for i in range(len(gather_list)):
            gather_list[i] = gather_list[i].numpy()
        res = []
        for i in range(len(gather_list)):
            for j in range(len(gather_list)):
                res.append(np.linalg.norm(gather_list[i]-gather_list[j]))
        df.loc[len(df)] = res
        df.to_csv(os.path.join(root, f'corssNorm.csv'))
    else:
        dist.gather(local_sigma, dst=0)
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
        p = mp.Process(target=init_process, args=(rank, size, trainloader[rank], run, 'gloo', 29566 if share_D else 29567))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
