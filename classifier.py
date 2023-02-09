# auto classifier
import os, sys, time, copy, types
from pathlib import Path

import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

model_path = Path(os.getcwd()).joinpath('classifier_models')
sys.path.append(Path(os.getcwd()).parent.absolute())
sys.path.append(Path(os.getcwd()).absolute())

import torch
import torch.nn as nn
from utils.datasets import celeba
from utils.loader import get_infinite_batches, load_dataset
from utils.loader import load_model as load_state_dict
from models.WGAN_GP import Generator, generate_images

import torch.distributed as dist
import torch.multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50', required=False)
parser.add_argument('--mode', type=str, default='saveData', required=False)
parser.add_argument('--load_model', type=str, default='False', required=False)
parser.add_argument('--num_epochs', type=int, default=24, required=False) # 13193*2/64
parser.add_argument('--share_D', type=str, default='True', required=False)

args = parser.parse_args()

model_name = args.model
mode = args.mode
load_model = True if args.load_model == 'True' else False
num_epochs = args.num_epochs
batch_size = 32
share_D = args.share_D
assert share_D in ['True', 'False']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


negative_proportion = [0.5, 0.1, 0.1, 0.1]

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def get_model_path():
    return str(model_path.joinpath(f'{model_name}' + '.pt'))
def setup_model():
    if model_name == 'resnet18':
        model = torchvision.models.resnet18()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == 'resnet50':
        model = resnet50(2, False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                                transform=transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])
                                ,proportion=0.5, 
                                rotary=False)
    trainset, valset = random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)])
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(valset, batch_size=64,
                        shuffle=True, drop_last=True)

def get_infinite_batches(data_loader):
    while True:
        for i, (images, flag )in enumerate(data_loader):
            yield images, flag

def train(model: torchvision.models.ResNet, optimizer: torch.optim.Optimizer, criterion = None, data = None, dataset_size = 0):
    assert data != None and criterion != None
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            for images, labels in data[phase]:
                # print("iter...")
                running_loss, running_corrects = 0, 0

                images = images.to(device)
                labels = labels.to(device)
                # print(labels)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if model_name == 'resnet18':
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1) # ?
                        loss = criterion(outputs, labels)
                    elif model_name == 'resnet50':
                        logits, preds = model(images)
                        preds = torch.argmax(preds, dim=1)
                        # print(logits.size(), probas.size(), labels.size())
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                        optimizer.zero_grad()
        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stat
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(data[phase])
            epoch_acc = running_corrects.double() / len(data[phase])

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

def eval_model(model: torchvision.models.ResNet, data = None, iters = 1000):
    assert data != None
    if isinstance(data, types.FunctionType):
        pass
    elif isinstance(data, Generator):
        positive, negative = 0, 0
        for iter in range(iters):
            batch = generate_images(data, 64, 0)
            for images in batch:
                outputs = model(images)
                _, preds = torch.max(outputs, 1) # ?
                for pred in preds:
                    if pred == 0:
                        negative += 1
                    elif pred == 1:
                        positive += 1
        ratio = positive / negative
        print(f"The positive negative ratio is: {positive}:{negative}, or {ratio}")

##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = torch.nn.functional.softmax(logits, dim=1)
        return logits, probas

def resnet50(num_classes, grayscale):
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model

def eval_model(GAN: Generator, model: ResNet, iters: int, device=0, batch_size=64):
    print('evaluating generator ratios...')
    posi_count ,num_examples = 0, 0
    for i in range(iters):
        if i % 10:
            print(f'iter: {i}|sampled pictures: {num_examples}')
        samples = generate_images(generator=GAN, batch_size=batch_size)
        # samples = torch.unbind(samples)
        samples = nn.functional.interpolate(samples, (128, 128)).to(device)
        logits, probas = model(samples)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += samples.size(0)
        posi_count += (predicted_labels == 1).sum()
    print(f'Running {iter} * {batch_size}:\nPositive count: {posi_count}\nNegative count {num_examples-posi_count}')
    print(f'ratio: {posi_count/(num_examples-posi_count)}:1')

def saveImageBatches(rank, size):
    device = rank
    path = os.path.join(os.getcwd(), 'samples' + 'share_D' + share_D + str(device))
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    GAN = Generator([3, 64, 64]).to(device)
    load_state_dict(f'runs/_WGAN-GP_0.4_{share_D}_CelebA_AvgMod_1_delay_/generator_pid_{device}.pkl', GAN)
    # saveImageBatches(GAN, str(path))
    for i in range(100):
        samples = generate_images(GAN, 64, device)
        samples = torch.unbind(samples, dim=0)
        for j in range(len(samples)):
            torchvision.utils.save_image(samples[j], path + "/" + str(i*64+j).zfill(6) + '.png')
            print(f'saving to: {path}/{str(i*64+j).zfill(6)}.png')

def saveData(rank, size):
    device = rank
    path = os.path.join(os.getcwd(), 'data' + str(device))
    if not os.path.exists(path):
        os.makedirs(path)
    trainloader, img_shape = load_dataset(dataset_name='CelebA',
                    random_colors='all_random', 
                    client_cnt=1, 
                    channels=3, 
                    batch_size=batch_size,
                    colors = None, 
                    debug=True,
                    root='.',
                    P_Negative=negative_proportion[rank])
    batch = 0
    for _, (image, _) in enumerate(trainloader[0]):
        image = torch.unbind(image)
        for j in range(len(image)):
            torchvision.utils.save_image(image[j], path + "/" + str(batch*64+j).zfill(6) + '.png')
            print(f'saving to: {path}/{str(batch*64+j).zfill(6)}.png')
        batch += 1

if __name__ == '__main__':
    if mode == 'train':
        model, optimizer, criterion = setup_model()
        trainloader, valloader = get_dataloader()
        data = {'train': trainloader, 'val': valloader}
        model = train(model=model, optimizer=optimizer, criterion=criterion, data=data)
        save_model(model)
    elif mode == 'GAN':
        # eval inputs from GAN.
        GAN = Generator([3, 64, 64]).to(0)
        load_state_dict('generator_pid_0.pkl', GAN)
        classifier = resnet50(2, False).to(0)
        load_state_dict('resnet50.pt', classifier)
        eval_model(GAN=GAN, model=classifier, iters=100, batch_size=64)
    elif mode == 'saveSamples':
        processes = []
        mp.set_start_method("spawn")
        size = torch.cuda.device_count()
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, saveImageBatches))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    
    elif mode == 'saveData':
        processes = []
        mp.set_start_method("spawn")
        size = torch.cuda.device_count()
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, saveData))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
