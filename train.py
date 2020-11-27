#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   用来训练模型
@File    :   train.py
@Time    :   2020/11/26 19:56:27
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/11/26 19:56:27
'''


import torch
from net import SRCNN
from torch import nn
import torch.optim as optim
from data import Generate_data
from utils import *
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    #GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    #set Model
    model = SRCNN().to(device)
    loss = nn.MSELoss()
    learn_rate = 1e-4
    optimizer = optim.Adam(
        model.parameters,
        lr=lr,
    )

    #Gerate_data
    train_paths, val_paths, test_paths = get_paths()

    HR_train = Generate_data(train_paths)
    LR_train = Generate_data(train_paths, mode='LR')

    HR_val = Generate_data(val_paths)
    LR_val = Generate_data(val_paths, mode='LR')

    HR_test = Generate_data(test_paths)
    LR_test = Generate_data(test_paths, mode='LR')