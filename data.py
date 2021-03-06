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
import gc



class Generate_data(Dataset):
    def __init__(self, paths, channels=31, fis=63, nums=10, down_size=[16,16], up_size=[63,63]):

        super(Generate_data, self).__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.HR = torch.zeros([len(paths), nums, channels, fis, fis])

        #Traverse all data
        for i in range(len(paths)):

            img = h5py.File(paths[i], 'r')['rad']
            img = np.array(img)
            img /= 4095.0
            img = torch.tensor(img)

            for num in range(nums):
                # 31 63 63
                crop_img = transforms.RandomCrop(63)(img)
                self.HR[i][num] = crop_img

        
        self.HR = self.HR.reshape((-1, channels, fis, fis)) 

        self.LR = interpolate(
                    self.HR,
                    size=down_size,
                    mode='bicubic'
                                )

        self.LR = interpolate(
                    self.LR,
                    size=up_size,
                    mode='bicubic'
                                )

        self.diff = torch.zeros([len(paths) * nums, channels, 3, fis, fis])
        for lr in range(self.LR.shape[0]):
            self.diff[lr] = self.diff_data(self.LR[lr])

        del self.LR
        gc.collect()

        self.HR = self.HR.reshape((-1, fis, fis))
        self.diff = self.diff.reshape((-1, 3, fis, fis))

    def __len__(self):
        return self.HR.shape[0]

    
    def __getitem__(self,index):
        return self.diff[index], self.HR[index]


    def diff_data(self,crop_img, channels=31, fis=63):
        res = torch.zeros(channels, 3, fis, fis)
        for l in range(channels -1):

                res[l][0] = crop_img[l]
                res[l][1] = crop_img[l] - crop_img[l + 1]
                res[l][2] = crop_img[l+1]

        res[channels - 1][0] = crop_img[channels - 1]
        res[channels - 1][2] = crop_img[channels - 1]

        return res


class Test_data(Dataset):

    def __init__(self, paths, channels=31, fis=63,down_size=[16,16], up_size=[63,63]):

        num = 252
        shape = [len(paths), num, channels, fis, fis]
        self.HR = torch.zeros(shape)

        for i in range(len(paths)):

            img = h5py.File(paths[i], 'r')['rad']
            img = np.array(img)
            img /= 4095.0
            img = torch.tensor(img)
        
            print(i,img.shape)
            count = 0
            for x in range(63 * 2, 1392 - 3 * fis, fis):
                for y in range(63 * 2, 1194 - 3 * fis, fis):
                    
                    self.HR[i][count] = img[:,x:x+fis,y:y+fis]
                    count += 1
            
        self.HR = self.HR.reshape((-1,channels,fis,fis))
                  

        self.LR = interpolate(
                    self.HR,
                    size=down_size,
                    mode='bicubic'
                                )

        self.LR = interpolate(
                    self.LR,
                    size=up_size,
                    mode='bicubic'
                                )

        self.diff = torch.zeros([len(paths) * num, channels, 3, fis, fis])
        for lr in range(self.LR.shape[0]):
            self.diff[lr] = self.diff_data(self.LR[lr])

        del self.LR
        gc.collect()

        self.HR = self.HR.reshape((-1, fis, fis))
        self.diff = self.diff.reshape((-1, 3, fis, fis))



    def __len__(self):
        return self.HR.shape[0]



    
    def __getitem__(self,index):
        return self.diff[index], self.HR[index]



    def diff_data(self,crop_img, channels=31, fis=63):
        res = torch.zeros(channels, 3, fis, fis)
        for l in range(channels -1):

                res[l][0] = crop_img[l]
                res[l][1] = crop_img[l] - crop_img[l + 1]
                res[l][2] = crop_img[l+1]

        res[channels - 1][0] = crop_img[channels - 1]
        res[channels - 1][2] = crop_img[channels - 1]

        return res
        
