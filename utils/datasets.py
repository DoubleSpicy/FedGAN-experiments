'''
custom dataset classes for loading different datasets
'''
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import random
import skimage.io as io
import torch.distributed as dist
import math

class celeba(Dataset):
    def __init__(self, root_dir: str, img_path: str, attr_data, attr_filter: list = None, transform: transforms = None, proportion: int = 0, rotary = True):
        self.root_dir = os.path.join(root_dir, __class__.__name__)
        self.img_path = os.path.join(self.root_dir, img_path)
        self.proportion = proportion
        print(self.root_dir, self.img_path)
        assert type(attr_data) in [str, pd.DataFrame]
        if type(attr_data) == str:
            self.attribute_data = self._load_csv(os.path.join(self.root_dir, attr_data), skip_first_row=True) # load from path
        else:
            self.attribute_data = attr_data # direct load
        self.attr_filter = attr_filter
        if rotary:
            drop_list = rotary_reduce(self.attribute_data.index.tolist())
            self.attribute_data.drop(drop_list)
        self._filter_attribute_tag()
        self.transform = transform

    def __len__(self):
        return len(self.attribute_data.index)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.img_path, self.attribute_data.iloc[index, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)
        tag = str(self.attr_filter[0]).lstrip('+').lstrip('-')
        flag = int(self.attribute_data.iloc[index][tag])
        flag = max(flag, 0)
        return image, flag

    def _process_tags(self):
        positive_list, negative_list = list(), list()
        for tags in self.attr_filter:
            if len(tags) > 0:
                if tags[0] == '-':
                    negative_list.append(tags.lstrip('+-'))
                else:
                    positive_list.append(tags.lstrip('+-'))
        return positive_list, negative_list

    def _filter_attribute_tag(self):
        remove_index_list = []
        positive_list, negative_list = self._process_tags()
        if self.proportion == 0:
            for tags in positive_list:
                remove_index_list += self.attribute_data[self.attribute_data[tags] != 1].index.to_list()
                # self.attribute_data.loc[self.attribute_data[tags] == -1] = 0
            for tags in negative_list:
                remove_index_list += self.attribute_data[self.attribute_data[tags] != -1].index.to_list()
        else:
            def prob_true(limit):
                return random.uniform(0, 1) <= limit
            # proportion = P(opposite class)

            if len(positive_list) == 1:
                # self.attribute_data.loc[self.attribute_data[tags] == -1] = 0
                tags = positive_list[0]
                negativeCnt = len(self.attribute_data[self.attribute_data[tags] == -1])
                positiveCnt = len(self.attribute_data) - negativeCnt
                dropCnt = math.floor((self.proportion * (positiveCnt + negativeCnt) - negativeCnt) / (self.proportion - 1))
                print(f'positiveCnt: {positiveCnt}, negativeCnt: {negativeCnt}, dropCnt: {dropCnt}')
                remove_index_list = np.random.choice(self.attribute_data[self.attribute_data[tags] == -1].index, dropCnt, replace=False)
                self.attribute_data = self.attribute_data.drop(remove_index_list)
        # print(self.attribute_data[self.attribute_data['Eyeglasses'] == 1])
        # print(self.attribute_data[self.attribute_data['Eyeglasses'] == -1])
        
        # print("rank:", dist.get_rank() if dist.is_initialized() else "0", len(self.attribute_data[self.attribute_data['Eyeglasses'] == 1]), len(self.attribute_data[self.attribute_data['Eyeglasses'] == -1]), len(self.attribute_data))


    def _load_csv(self, path, skip_first_row=False):
        rows_to_skip = []
        if skip_first_row:
            rows_to_skip.append(0)
        df = pd.read_csv(path, sep=" ", skiprows=rows_to_skip)
        df.columns = ["filename"] + df.columns.tolist()[:-1]
        return df

def rotary_reduce(target_list: list):
    drop_list = []
    pid, size = dist.get_rank(), dist.get_world_size()
    cnt = 0
    for i in target_list:
        if cnt != pid:
            drop_list.append(i)
            cnt += 1
        else:
            cnt -= size
    return drop_list

def equalize(a: celeba, b: celeba):
    # input 2 datasets, equalize their size to min of the them.
    size = min(len(a.attribute_data), len(b.attribute_data))
    def drop_rows_randomly(dataset: celeba):
        to_remove = len(dataset.attribute_data) - size
        drop_indices = np.random.choice(dataset.attribute_data.index, to_remove, replace=False)
        dataset.attribute_data = dataset.attribute_data.drop(drop_indices)
    drop_rows_randomly(a)
    drop_rows_randomly(b)

def equalizePD(a: pd.DataFrame, tag: str, proportion: float):
    positiveCnt, negativeCnt = len(a[a[tag] == 1]), len(a[a[tag] == -1])
    to_remove = math.floor(negativeCnt - proportion * positiveCnt)
    # (+veCnt / (-veCnt - x)) = proportion.
    # +veCnt / proportion + -veCnt
    drop_indices = np.random.choice(a[a[tag] == -1].index, to_remove, replace=False)
    a = a.drop(drop_indices)
    return a

# if __name__ == '__main__':
#     dataset = celeba(root_dir='../data/', attr_data='list_attr_celeba.txt', img_path='img_align_celeba', attr_filter=['5_o_Clock_Shadow', '-Arched_Eyebrows'])
#     print(dataset.attribute_data)
#     print(len(dataset.attribute_data[dataset.attribute_data['Eyeglasses'] == 1]))
#     print(len(dataset.attribute_data[dataset.attribute_data['Eyeglasses'] == -1]))
#     print(len(dataset.attribute_data[dataset.attribute_data['Mustache'] == 1]))
#     print(len(dataset.attribute_data[dataset.attribute_data['Mustache'] == -1]))
#     # img = dataset.__getitem__(index=0)
#     # print(dataset.attribute_data)
#     # img.save('test.jpg')


class TinyImageNet(Dataset):
    def __init__(self, root_dir: str, img_path: str, attr_data, attr_filter: list = None, transform: transforms = None):
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir, img_path)
        print(self.root_dir, self.img_path)
        assert type(attr_data) in [str, pd.DataFrame]
        if type(attr_data) == str:
            self.attribute_data = self._load_csv(os.path.join(self.root_dir, attr_data), skip_first_row=False) # load from path
            # print(self.attribute_data)
        else:
            self.attribute_data = attr_data # direct load
        self.attr_filter = list()
        if attr_filter is not None:
            self.attr_filter += attr_filter
        self._filter_attribute_tag()
        self.transform = transform

    def _load_csv(self, path, skip_first_row=False):
        rows_to_skip = []
        if skip_first_row:
            rows_to_skip.append(0)
        df = pd.read_csv(path, skiprows=rows_to_skip)
        return df

    def _process_tags(self):
        positive_list, negative_list = list(), list()
        for tags in self.attr_filter:
            if len(tags) > 0:
                if tags[0] == '-':
                    negative_list.append(tags.lstrip('+-'))
                else:
                    positive_list.append(tags.lstrip('+-'))
        return positive_list, negative_list

    def _filter_attribute_tag(self):
        if len(self.attr_filter) == 0:
            return
        remove_index_list = []
        positive_list, negative_list = self._process_tags()
        # print(positive_list, negative_list)
        for i in range(self.attribute_data.shape[0]):
            splitted = set(self.attribute_data.iloc[i]['tags'].split(', '))
            if(len(splitted.intersection(positive_list)) == 0 or len(splitted.intersection(negative_list)) > 0):
                remove_index_list.append(i)
        # if len(positive_list) > 0:
        #     remove_index_list += self.attribute_data[~self.attribute_data['tags'].str.contains('|'.join(positive_list), na=False)].index.to_list()
        # print('len:', len(remove_index_list))
        # if len(negative_list) > 0:
        #     remove_index_list += self.attribute_data[self.attribute_data['tags'].str.contains('|'.join(negative_list), na=False)].index.to_list()
        # print('len:', len(remove_index_list))
        self.attribute_data = self.attribute_data.drop(remove_index_list)
        drop_list = rotary_reduce(self.attribute_data.index.tolist())
        self.attribute_data = self.attribute_data.drop(drop_list)
        print(self.attribute_data)
        print(self.attribute_data['tags'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        class_id = self.attribute_data.iloc[index // 500]['id']
        class_path = os.path.join(self.img_path, class_id + '/images')
        img_name = class_id + '_' + str(index % 500) + '.JPEG'
        img_path = os.path.join(class_path, img_name)
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            # print(image.shape)
        else:
            image = np.array(image)
        return image, self.attr_filter



    def __len__(self):
        return len(self.attribute_data.index)*500


class CelebA(Dataset):
    def __init__(self, root='../data/', tags: list = None, transform: transforms = None) -> None:
        super().__init__()
        self.root_dir = os.path.join(root, 'celeba')
        self.img_path = os.path.join(self.root_dir, 'img_align_celeba')
        self.tags = tags
        self.attribute_data = pd.read_csv(os.path.join(self.root_dir, 'list_attr_celeba.txt'), sep=' ', skiprows=[0])
        self.attribute_data.columns = ["filename"] + self.attribute_data.columns.tolist()[:-1]
        self.attribute_data = self.attribute_data[['filename'] + tags]
        self.transform = transform
        print(self.attribute_data)
    
    def load(self, df: pd.DataFrame):
        self.attribute_data = df
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.img_path, self.attribute_data.iloc[index, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)
        flag = int(self.attribute_data.iloc[index][self.tags[0]])
        flag = max(flag, 0)
        return image, flag

    def __len__(self):
        return len(self.attribute_data.index)


def splitCelebA(df: pd.DataFrame, client_ratio: list, tag: list = None) -> list:
    tag = tag[0]
    client_positive, client_negative = 0.0, 0.0
    for i in range(len(client_ratio)):
        client_positive += float(client_ratio[i][0])
        client_negative += float(client_ratio[i][1])
    glob_positive, glob_negative = 1, client_negative/ client_positive 
    positive, negative = len(df[df[tag] == 1]), len(df[df[tag] == -1])
    # normalize global_ratio
    # calculate how many items to get in list
    flip = positive if glob_negative >= float(negative)/float(positive) else negative # the one with lesser then required
    if glob_negative >= float(negative)/float(positive): # not enough negative samples
        positive_filenames = np.random.choice(df[df[tag] == 1]['filename'].tolist(), int(negative/glob_negative), replace=False)
        negative_filenames = np.random.choice(df[df[tag] == -1]['filename'].tolist(), negative, replace=False)
    else: # not enough positive samples
        positive_filenames = np.random.choice(df[df[tag] == 1]['filename'].tolist(), positive, replace=False)
        negative_filenames = np.random.choice(df[df[tag] == -1]['filename'].tolist(), int(positive*glob_negative), replace=False)
    
    # total_len = len(positive_index) + len(negative_index)
    # calculate splitting to each one
    positive_index = df[df['filename'].isin(positive_filenames)].index
    negative_index = df[df['filename'].isin(negative_filenames)].index
    client_positive = [int(len(positive_index) * ratio[0] / (client_positive + 1e-9)) for ratio in client_ratio]
    client_negative = [int(len(negative_index) * ratio[1] / (client_negative + 1e-9)) for ratio in client_ratio]
    prevPos, prevNeg = 0, 0
    client_positive_index, client_negative_index = [], []
    for i in range(len(client_ratio)):
        client_positive_index.append(positive_index[prevPos: prevPos + client_positive[i]+1])
        client_negative_index.append(negative_index[prevNeg: prevNeg + client_negative[i]+1])
        prevPos += client_positive[i]+1
        prevNeg += client_negative[i]+1

    return [df[df.index.isin(a) | df.index.isin(b)] for a, b in zip(client_positive_index, client_negative_index)]
    

# if __name__ == '__main__':
#     data0 = CelebA(root='../data/', tags=['Eyeglasses'])
#     data = split(data0.attribute_data, client_ratio=[[0.5, 0.5],  [0.9, 0.1],  [0.9, 0.1], [0.9, 0.1]], tag=['Eyeglasses'])
#     print('===============')
#     for d in data:
#         print(d)
#         print(len(d[d['Eyeglasses'] == 1]), len(d[d['Eyeglasses'] == -1]))

#     dataCifar = torchvision.datasets.CIFAR10('../data/', download=True)
#     print(dataCifar.targets)