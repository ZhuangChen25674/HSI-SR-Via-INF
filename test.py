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
from net import INF
import torch
from data import Test_data
from torch.utils.data import DataLoader



TEST_BATCH_SIZE = 31
WEIGHT_PATH = '/home/yons/data1/chenzhuang/HSI-SR/HSI-SR-Via-INF/weight/'
PSNRS = []
SAMS = []
PSNR_IMG = torch.zeros([252*8, 31, 63, 63])
SAM_IMG = torch.zeros([252*8, 31, 63, 63])
HR = torch.zeros([252*8, 31, 63, 63])

if __name__ == "__main__":
    
    _, _, test_paths = get_paths()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test decice is {}'.format(device))

    #best psnr model
    model_psnr = INF().to(device)
    state_dict_psnr = model_psnr.state_dict()

    #best sam model
    model_sam = INF().to(device)
    state_dict_sam = model_sam.state_dict()

    psnr_path = WEIGHT_PATH + 'INF_best_psnr.pth'
    # load all tensor to gpu 1
    for n, p in torch.load(psnr_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict_psnr.keys():
            state_dict_psnr[n].copy_(p)
        else:
            raise KeyError(n)

    sam_path = WEIGHT_PATH + 'INF_best_sam.pth'
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
            #保存psnr结果
            PSNR_IMG[count-1] = psnr_preds

            sam_preds = model_sam(
                    lr[:,0,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,1,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    lr[:,2,:,:].reshape((TEST_BATCH_SIZE,1,63,63)),
                    )

            sam_preds = sam_preds.reshape((TEST_BATCH_SIZE,63,63))
            #保存sam结果
            SAM_IMG[count-1] = sam_preds

            HR[count-1] = hr
            psnr_pre_name = "pre_img{}_{}_psnr_model_psnr{}_sam{}.png".format(
                count//252,
                count % 252,
                calc_psnr(psnr_preds,hr),
                SAM_GPU(psnr_preds,hr)
            )
            sam_pre_name = 'pre_img{}_{}_sam_model_psnr{}_sam{}.png'.format(
                count//252,
                count % 252,
                calc_psnr(sam_preds,hr),
                SAM_GPU(sam_preds,hr)
            )

            hr_name = 'hr_img{}_{}.png'.format(
                count//252,
                count % 252,
            )

        # print('img{}_{}_psnr_model : test psnr is {:.2f}, test sam is {:.2f}'.format(
        #         count//252,
        #         count % 252,
        #         calc_psnr(psnr_preds,hr),
        #         SAM_GPU(psnr_preds,hr)
        # ))


        # print('img{}_{}_sam_model : test psnr is {:.2f}, test sam is {:.2f}'.format(
        #         count//252,
        #         count % 252,
        #         calc_psnr(sam_preds,hr),
        #         SAM_GPU(sam_preds,hr)
        # ))


        PSNRS.append((calc_psnr(psnr_preds,hr),calc_psnr(sam_preds,hr)))
        SAMS.append((SAM_GPU(psnr_preds,hr),SAM_GPU(sam_preds,hr)))


        if count % 252 == 0:
            begin = int((count / 252)) - 1
            begin *= 252
            end =  int((count / 252))
            end *= 252

            
            print('psnr model img{} average psnr is {:.2f} average sam is {:.2f} \
max_min psnr is {:.2f} {:.2f} max_min sam is {:.2f} {:.2f}'.format(
                int(count / 252),
                sum([i[0]  for i in PSNRS[begin:end]])/ 252.0,
                sum([i[0]  for i in SAMS[begin:end]]) / 252.0,
                max([i[0]  for i in PSNRS[begin:end]]),
                min([i[0]  for i in PSNRS[begin:end]]),
                max([i[0]  for i in SAMS[begin:end]]),
                min([i[0]  for i in SAMS[begin:end]])
            ))

            print('sam model img{} average psnr is {:.2f} average sam is {:.2f} \
max_min psnr is {:.2f} {:.2f} max_min sam is {:.2f} {:.2f}'.format(
                int(count / 252),
                sum([i[1] for i in PSNRS[begin:end]]) / 252.0,
                sum([i[1] for i in SAMS[begin:end]]) / 252.0,
                max([i[1]  for i in PSNRS[begin:end]]),
                min([i[1]  for i in PSNRS[begin:end]]),
                max([i[1]  for i in SAMS[begin:end]]),
                min([i[1]  for i in SAMS[begin:end]])
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

    # torch.save(PSNR_IMG, '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/PSNR_IMG.pth')
    # torch.save(SAM_IMG, '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/SAM_IMG.pth')
    # torch.save(HR, '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/HR.pth')
    # print(count)