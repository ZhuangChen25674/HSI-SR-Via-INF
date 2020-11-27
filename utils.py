#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   工具函数
@File    :   utils.py
@Time    :   2020/11/26 20:15:29
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/11/26 20:15:29
'''


from data import Generate_data
from torch.utils.data import DataLoader

PATH = 'DataSet/ICVL/'

#return all paths
def get_paths(PATH = PATH):

    train_paths = []
    val_paths = []
    test_paths = []

    with open(PATH + 'train_names.txt', 'r') as f:
        for i in f.readlines():
            train_paths.append(PATH + i.strip())

    with open(PATH + 'val_names.txt', 'r') as f:
        for i in f.readlines():
            val_paths.append(PATH + i.strip())

    with open(PATH + 'test_names.txt', 'r') as f:
        for i in f.readlines():
            test_paths.append(PATH + i.strip())

    return train_paths, val_paths, test_paths


def get_dataloader(paths):
    
    HR_data = Generate_data(paths)
    LR_data = Generate_data(paths, mode='LR')

    HR_dataloader = DataLoader(
        HR_data,
        batch_size= 16,
        shuffle=True,
        num_workers= 2, 
        pin_memory= True,
        drop_last= True,
    )
    LR_dataloader = DataLoader(
        LR_data,
        batch_size= 16,
        shuffle=True,
        num_workers= 2, 
        pin_memory= True,
        drop_last= True,
    )

    return HR_dataloader, LR_dataloader
