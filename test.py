#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   测试模型性能
@File    :   test.py
@Time    :   2020/12/04 15:38:51
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/04 15:38:51
'''

from utils import *
from net import SRCNN
import torch
from data import Test_data
from torch.utils.data import DataLoader



TEST_BATCH_SIZE = 31
WEIGHT_PATH = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/weight/'
PSNRS = []
SAMS = []


if __name__ == "__main__":
    
    _, _, test_paths = get_paths()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test decice is {}'.format(device))

    #best psnr model
    model_psnr = SRCNN().to(device)
    state_dict_psnr = model_psnr.state_dict()

    #best sam model
    model_sam = SRCNN().to(device)
    state_dict_sam = model_sam.state_dict()

    psnr_path = WEIGHT_PATH + 'best_psnr.pth'
    # load all tensor to gpu 1
    for n, p in torch.load(psnr_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict_psnr.keys():
            state_dict_psnr[n].copy_(p)
        else:
            raise KeyError(n)

    sam_path = WEIGHT_PATH + 'best_sam.pth'
    # load all tensor to gpu 1
    for n, p in torch.load(sam_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict_sam.keys():
            state_dict_sam[n].copy_(p)
        else:
            raise KeyError(n)


    model_psnr.eval()
    
    HR_data = Test_data(test_paths)

    test_data = DataLoader(
                HR_data,
                batch_size=TEST_BATCH_SIZE,
                shuffle=False,
                num_workers= 2, 
                pin_memory= True,
                drop_last= True,
            )

    count = 1
    for lr, hr in test_data:

        lr = lr.to(device)
        hr = hr.to(device)

        with torch.no_grad():
            psnr_preds = model_psnr(
                    lr[:,0,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,1,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,2,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    )

            psnr_preds = psnr_preds.reshape((TEST_BATCH_SIZE,63,63))

            sam_preds = model_sam(
                    lr[:,0,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,1,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,2,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    )

            sam_preds = sam_preds.reshape((TEST_BATCH_SIZE,63,63))

        print('img{}_psnr_model : test psnr is {:.2f}, test sam is {:.2f}'.format(
                count,
                calc_psnr(psnr_preds,hr),
                SAM_GPU(psnr_preds,hr)
        ))


        print('img{}_sam_model : test psnr is {:.2f}, test sam is {:.2f}'.format(
                count,
                calc_psnr(sam_preds,hr),
                SAM_GPU(sam_preds,hr)
        ))


        PSNRS.append((calc_psnr(psnr_preds,hr),calc_psnr(sam_preds,hr)))
        SAMS.append((SAM_GPU(psnr_preds,hr),SAM_GPU(sam_preds,hr)))


        if count % 396 == 0:
            begin = int((count / 396)) - 1
            begin *= 396
            end =  int((count / 396))
            end *= 396
            # print(begin,end)
            # print(PSNRS[begin])
            print(sum([i[0]  for i in PSNRS[begin:end]]))
            print('psnr model img{} average psnr is {:.2f} average sam is {:.2f}'.format(
                int(count / 396),
                sum([i[0]  for i in PSNRS[begin:end]])/ 396.0,
                sum([i[0]  for i in SAMS[begin:end]]) / 396.0,
            ))

            print('sam model img{} average psnr is {:.2f} average sam is {:.2f}'.format(
                int(count / 396),
                sum([i[1] for i in PSNRS[begin:end]]) / 396.0,
                sum([i[1] for i in SAMS[begin:end]]) / 396.0,
            ))

        count += 1

    print('averge psnr of psnr model is {:.2f}'.format(
        sum([i[0] for i in PSNRS]) / len(PSNRS)
    ))
    print('averge sam of psnr model is {:.2f}'.format(
        sum([i[0] for i in SAMS]) / len(SAMS)
    ))

    print('averge psnr of sam model is {:.2f}'.format(
        sum([i[1] for i in PSNRS]) / len(PSNRS)
    ))
    print('averge sam of sam model is {:.2f}'.format(
        sum([i[1] for i in SAMS]) / len(SAMS)
    ))