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
import numpy as np
from PIL import Image


PATH = '/home/hefeng/data1/HSI-SR/DataSet/ICVL/'


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


def SAM_GPU(im_fake, im_true):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


def plot():

    from matplotlib import pyplot as plt

    path = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/train.log'
    sams = []
    psnrs = []

    with open(path,'r') as f:
        for line in f.readlines():
            
            line = line.strip()

            if 'average sam' in line:
                sams.append(float(line.split(' ')[-1]))

            if 'average psnr' in line:
                psnrs.append(float(line.split(' ')[-1]))

    epochs = [i for i in range(400)]

    fib_size = (5,4)
    fon_size = 12

    plt.figure(figsize=fib_size)
    plt.title('sam of every epoch',fontsize=fon_size)
    plt.xlabel('epoch',fontsize=fon_size)
    plt.ylabel('sam', fontsize=fon_size)
    plt.plot(epochs, sams, 'k.')
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1.1")
    plt.savefig('sam.tiff',dpi=600,format='tiff')


    plt.figure(figsize=fib_size)
    plt.title('psnr of every epoch',fontsize=fon_size)
    plt.xlabel('epoch',fontsize=fon_size)
    plt.ylabel('psnr', fontsize=fon_size)
    plt.plot(epochs, psnrs, 'k.')
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1.1")
    plt.savefig('psnr.tiff',dpi=600,format='tiff')


def show_img(img, name):

    shape = img.shape
    img = img.to('cpu')
    img = img.numpy()

    b = np.mean(img[:11], axis=0)
    g = np.mean(img[11:21], axis=0)
    r = np.mean(img[21:], axis=0)

    rgb = np.zeros((shape[1],shape[2],3))
    rgb[: ,: ,0] = r
    rgb[: ,: ,1] = g
    rgb[: ,: ,2] = b
    rgb *= 255
    rgb = rgb.astype(np.uint8)

    img = Image.fromarray(rgb)

    path = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/' + name 
    
    img.save(path)

