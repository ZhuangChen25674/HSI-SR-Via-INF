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


from torch.nn.functional import interpolate
import torch

PATH = '/home/hefeng/data1/HSI-SR/DataSet/ICVL/'

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


#return all paths
def get_paths(PATH = PATH):

    train_paths = []
    val_paths = []
    test_paths = []

    with open(PATH + 'train_name.txt', 'r') as f:
        for i in f.readlines():
            train_paths.append(PATH + i.strip())

    with open(PATH + 'val_name.txt', 'r') as f:
        for i in f.readlines():
            val_paths.append(PATH + i.strip())

    with open(PATH + 'test_name.txt', 'r') as f:
        for i in f.readlines():
            test_paths.append(PATH + i.strip())

    return train_paths, val_paths, test_paths


def get_LR(data,down_size=[16,16], up_size=[63,63]):
    # num 3 63 63
    data = interpolate(
                    data,
                    size=down_size,
                    mode='bicubic'
                                )

    data = interpolate(
                    data,
                    size=up_size,
                    mode='bicubic'
                                )
    return data



def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count