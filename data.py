#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   生成训练 验证 测试数据集
@File    :   diff_model.py
@Time    :   2020/11/25 16:18:22
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/11/25 16:18:22
'''

from torch.utils.data import Dataset
import h5py
import torch
from torchvision import transforms 
from torch.nn.functional import interpolate
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Generate_data(Dataset):
    def __init__(self, paths,  mode='HR', channels=31, fis=63,down_size=[16,16], up_size=[63,63]):

        super(Generate_data, self).__init__()

        self.mode = mode
        self.down_size = down_size
        self.up_size = up_size
        self.shape = [len(paths), (channels-1), 3, fis, fis] #slice = 3
        self.data = torch.zeros(self.shape)

        #Traverse all data
        for i in range(len(paths)):

            img = h5py.File(paths[i], 'r')['rad']
            img = np.array(img)
            img = img / (np.max(img) - np.min(img))
            img = torch.tensor(img)

            img = transforms.RandomCrop(63)(img)

            # diff data
            for l in range(channels - 1):

                self.data[i][l][0] = img[l]
                self.data[i][l][1] = img[l + 1]
                self.data[i][l][2] = img[l+1] - img[l]


        self.data = self.data.reshape((-1, 3, fis, fis))
        self.shape = self.data.shape


        if self.mode == 'LR':
            #downsample
            self.data = interpolate(self.data, 
                                size=self.down_size,
                                mode='bicubic'
                                )
        
            #upsample
            self.data = interpolate(
                                    self.data,
                                    size=self.up_size,
                                    mode='bicubic'
                                )


    def __len__(self):
        return self.shape[0] 
    

    def __getitem__(self, index):
        return self.data[index]
        