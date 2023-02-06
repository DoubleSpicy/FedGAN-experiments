# auto classifier
import os, sys, time, copy
from pathlib import Path

import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

model_path = Path(os.getcwd()).joinpath('classifier_models')
sys.path.append(Path(os.getcwd()).parent.absolute())
sys.path.append(Path(os.getcwd()).absolute())

import torch
from utils.datasets import celeba
from utils.loader import get_infinite_batches






parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18', required=False)
parser.add_argument('--mode', type=str, default='train', required=False)
parser.add_argument('--load_model', type=str, default='False', required=False)
parser.add_argument('--num_epochs', type=int, default=1000, required=False) # 13193*2/64
args = parser.parse_args()

model_name = args.model
mode = args.mode
load_model = True if args.load_model == 'True' else False
num_epochs = args.num_epochs
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_path():
    return str(model_path.joinpath(f'{model_name}' + '.pt'))
def setup_model():
    if model_name == 'resnet18':
        model = torchvision.models.resnet18()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
    if load_model:
        weights = torch.load(get_model_path())
        model.load_state_dict(weights)

    return model.to(device), optimizer, criterion

def save_model(model: torchvision.models.ResNet):
    path = get_model_path()
    torch.save(model.state_dict(), path)
    print(f'{model_name} saved to {get_model_path}')

def get_dataloader():
    # achtung! output: [64, 3, 64, 64]
    dataset = celeba(root_dir='../data/', 
                                attr_data='list_attr_celeba.txt', 
                                img_path='img_align_celeba', 
                                attr_filter=['+Eyeglasses'],
                                transform=transforms.Compose([transforms.Resize([64, 64]),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5, ), (0.5, ))])
                                ,proportion=0.5, 
                                rotary=False)
    return DataLoader(dataset, batch_size=64,
                        shuffle=True, drop_last=True)

def get_infinite_batches(data_loader):
    while True:
        for i, (images, flag )in enumerate(data_loader):
            yield images, flag

def train(model: torchvision.models.ResNet, optimizer: torch.optim.Optimizer, criterion = None, data = None):
    assert data != None and criterion != None
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        images, labels = data.__next__()
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0, 0

            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(images)
                _, preds = torch.max(outputs, 1) # ?
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # stat
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / 64
        epoch_acc = running_corrects.double() / 64

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
if __name__ == '__main__':
    model, optimizer, criterion = setup_model()
    data_loader = get_dataloader()
    model = train(model=model, optimizer=optimizer, criterion=criterion, data=get_infinite_batches(data_loader))
    save_model(model)

