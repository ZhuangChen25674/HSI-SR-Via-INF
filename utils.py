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


PATH = '/home/yons/data1/chenzhuang/HSI-SR/DataSet/ICVL/'


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


def show_img(fis=63):

    PSNR_IMG = torch.load('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/PSNR_IMG.pth')
    SAM_IMG = torch.load('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/SAM_IMG.pth')

    PSNR_IMG = PSNR_IMG.reshape([8, 252, 31, 63, 63])
    SAM_IMG = SAM_IMG.reshape([8, 252, 31, 63, 63])

    for i in range(8):
        psnr_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        sam_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        hr_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        count = 0

        for x in range(0, 1134 , fis):
            for y in range(0, 882 , fis):
                psnr_img[:,x:x+fis,y:y+fis] = PSNR_IMG[i][count]
                sam_img[:,x:x+fis,y:y+fis] = SAM_IMG[i][count]
                count += 1
        
        psnr_img = psnr_img.numpy()
        b = np.mean(psnr_img[:11], axis=0)
        print(b.shape)
        g = np.mean(psnr_img[11:21], axis=0)
        r = np.mean(psnr_img[21:], axis=0)

        rgb = np.zeros((1134,882,3))
        rgb[: ,: ,0] = r
        rgb[: ,: ,1] = g
        rgb[: ,: ,2] = b
        rgb *= 255
        rgb = rgb.astype(np.uint8)

        img = Image.fromarray(rgb)
        img = img.rotate(180)
        
        path = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/psnr_img{}.png'.format(
            i
        )
    
        img.save(path)

        sam_img = sam_img.numpy()
        b = np.mean(sam_img[:11], axis=0)
        g = np.mean(sam_img[11:21], axis=0)
        r = np.mean(sam_img[21:], axis=0)

        rgb = np.zeros((1134,882,3))
        rgb[: ,: ,0] = r
        rgb[: ,: ,1] = g
        rgb[: ,: ,2] = b
        rgb *= 255
        rgb = rgb.astype(np.uint8)

        img = Image.fromarray(rgb)
        img = img.rotate(180)
        path = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/sam_img{}.png'.format(
            i
        )
    
        img.save(path)

def save_img(fis=63):

    PSNR_IMG = torch.load('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/PSNR_IMG.pth')
    SAM_IMG = torch.load('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/SAM_IMG.pth')
    HR = torch.load('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/HR.pth')

    PSNR_IMG = PSNR_IMG.reshape([8, 252, 31, 63, 63])
    SAM_IMG = SAM_IMG.reshape([8, 252, 31, 63, 63])
    HR = HR.reshape([8, 252, 31, 63, 63])

    for i in range(8):
        psnr_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        sam_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        hr_img = torch.zeros([31, 18 * 63, 14 * 63, ])
        count = 0

        for x in range(0, 1134 , fis):
            for y in range(0, 882 , fis):
                psnr_img[:,x:x+fis,y:y+fis] = PSNR_IMG[i][count]
                sam_img[:,x:x+fis,y:y+fis] = SAM_IMG[i][count]
                hr_img[:,x:x+fis,y:y+fis] = HR[i][count]
                count += 1
        print(count)
        psnr_img = psnr_img.numpy()
        psnr_img = np.transpose(psnr_img, (1, 2, 0))
        psnr_img *= 255
        psnr_img = psnr_img.astype(np.uint8)


        sam_img = sam_img.numpy()
        sam_img = np.transpose(sam_img,(1,2,0))
        sam_img *= 255
        sam_img = sam_img.astype(np.uint8)

        hr_img = hr_img.numpy()
        hr_img = np.transpose(hr_img,(1,2,0))
        hr_img *= 255
        hr_img = hr_img.astype(np.uint8)
        print(i, psnr_img.shape,sam_img.shape)
        
        np.save('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/psnr_img{}.npy'.format(i+1), psnr_img)
        np.save('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/sam_img{}.npy'.format(i+1), sam_img)
        # np.save('/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/img/hr_img{}.npy'.format(i+1), hr_img)


