'''
custom dataset classes for loading different datasets
'''
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import random
import skimage.io as io

class celeba(Dataset):
    def __init__(self, root_dir: str, img_path: str, attr_data, attr_filter: list = None, transform: transforms = None):
        self.root_dir = os.path.join(root_dir, __class__.__name__)
        self.img_path = os.path.join(self.root_dir, img_path)
        print(self.root_dir, self.img_path)
        assert type(attr_data) in [str, pd.DataFrame]
        if type(attr_data) == str:
            self.attribute_data = self._load_csv(os.path.join(self.root_dir, attr_data), skip_first_row=True) # load from path
        else:
            self.attribute_data = attr_data # direct load
        self.attr_filter = attr_filter
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
        return image, self.attr_filter

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
        for tags in positive_list:
            remove_index_list += self.attribute_data[self.attribute_data[tags] != 1].index.to_list()
        for tags in negative_list:
            remove_index_list += self.attribute_data[self.attribute_data[tags] != -1].index.to_list()
        # print(remove_index_list)
        self.attribute_data = self.attribute_data.drop(remove_index_list)

    def _load_csv(self, path, skip_first_row=False):
        rows_to_skip = []
        if skip_first_row:
            rows_to_skip.append(0)
        df = pd.read_csv(path, sep=" ", skiprows=rows_to_skip)
        df.columns = ["filename"] + df.columns.tolist()[:-1]
        return df

# if __name__ == '__main__':
    # dataset = celeba(root_dir='../data/', attr_data='list_attr_celeba.txt', img_path='img_align_celeba', attr_filter=['5_o_Clock_Shadow', '-Arched_Eyebrows'])
    # img = dataset.__getitem__(index=0)
    # print(dataset.attribute_data)
    # img.save('test.jpg')