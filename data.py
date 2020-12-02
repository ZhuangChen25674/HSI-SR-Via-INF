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




class Generate_data(Dataset):
    def __init__(self, paths, channels=31, fis=63, nums=10):

        super(Generate_data, self).__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        shape = [len(paths), nums, channels, 3, fis, fis]
        self.data = torch.zeros(shape)
        self.HR = torch.zeros([len(paths), nums, channels, fis, fis])

        #Traverse all data
        for i in range(len(paths)):

            img = h5py.File(paths[i], 'r')['rad']
            img = np.array(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = torch.tensor(img)

            for num in range(nums):
                # 31 63 63
                crop_img = transforms.RandomCrop(63)(img)
                self.HR[i][num] = crop_img

                # 31 3 63 63
                self.data[i][num] = self.diff_data(crop_img)


        self.data = self.data.reshape((-1, 3 ,fis, fis))
        self.HR = self.HR.reshape((-1, fis, fis))

    def __len__(self):
        return self.data.shape[0]

    
    def __getitem__(self,index):
        return self.data[index]


    def diff_data(self,crop_img, channels=31, fis=63):
        res = torch.zeros(channels, 3, fis, fis)
        for l in range(channels -1):

                res[l][0] = crop_img[l]
                res[l][1] = crop_img[l] - crop_img[l + 1]
                res[l][2] = crop_img[l+1]

        res[channels - 1][0] = crop_img[channels - 1]
        res[channels - 1][2] = crop_img[channels - 1]

        return res


    def get_HR(self):
        return self.HR

    def size(self, index):
        return self.data.size(index)
    

        

        
