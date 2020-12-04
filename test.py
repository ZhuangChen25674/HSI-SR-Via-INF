import h5py
import torchvision.transforms as t
import torch
import numpy as np


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# PATH = '/home/hefeng/data1/HSI-SR/DataSet/ICVL/'
# name = '4cam_0411-1648.mat'

# img = h5py.File(PATH + name, 'r')
# data = img['rad'] # 31 1392 1300
# data1 = np.array(data)


# loss_fn = torch.nn.MSELoss(reduction='sum') 
# input = torch.autograd.Variable(torch.randn(3,4)) 
# target = torch.autograd.Variable(torch.randn(3,4)) 
# loss = loss_fn(input, target) 
# print(input); print(target); print(loss) 
# print(input.size(), target.size(), loss.size())





a = torch.rand((31,63,63))
b = torch.rand((31,63,63))

print(SAM_GPU(a,b))