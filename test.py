import h5py
import torchvision.transforms as t
import torch
import numpy as np


torch.manual_seed(0)
torch.cuda.manual_seed(0)
PATH = '/home/hefeng/data1/HSI-SR/DataSet/ICVL/'
name = '4cam_0411-1648.mat'

img = h5py.File(PATH + name, 'r')
data = img['rad'] # 31 1392 1300
data1 = np.array(data)




