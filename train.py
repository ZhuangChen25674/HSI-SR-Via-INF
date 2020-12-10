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
from torch.utils.data import DataLoader, TensorDataset
import os, copy


EPOCHS = 400
LR = 1e-4
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 31
OUT_DIR = '/home/hefeng/data1/HSI-SR/HSI-SR-Via-INF/weight/'

if __name__ == "__main__":
    
    #GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is {}'.format(device))
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    #set Model
    model = SRCNN().to(device)
    criterion = nn.MSELoss(reduction='sum')

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    #Gerate_data
    train_paths, val_paths, _ = get_paths()


    best_weights = copy.deepcopy(model.state_dict())
    best_weights_sam = copy.deepcopy(model.state_dict())

    best_epoch = 0
    best_epoch_sam = 0
    best_psnr = 0.0
    best_sam = 180.0

    for epoch in range(EPOCHS):

        model.train()
        epoch_losses = AverageMeter()

        print('epoch:{}/{}'.format(epoch ,EPOCHS))

        HR_train_paths = DataLoader(
        train_paths,
        batch_size=4,
        shuffle=True,
    )

        for paths in HR_train_paths:
            
            # _ 31 63 63
            HR_data = Generate_data(paths)

            train_data = DataLoader(
                HR_data,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers= 2, 
                pin_memory= True,
                drop_last= True,
            )
            
            count = 0
            for lr, hr in train_data:
                #lr BS 3 63 63
                #hr BS 63 63

                count += 1
                lr = lr.to(device)
                hr = hr.to(device)

                preds = model(
                    lr[:,0,:,:].reshape((TRAIN_BATCH_SIZE,1,63,63)),
                    lr[:,1,:,:].reshape((TRAIN_BATCH_SIZE,1,63,63)),
                    lr[:,2,:,:].reshape((TRAIN_BATCH_SIZE,1,63,63)),
                )

                hr = hr.reshape((TRAIN_BATCH_SIZE, 1, 63, 63))

                loss = criterion(preds, hr)

                epoch_losses.update(loss.item(), len(lr))
                
                if count % 4 == 0:
                    print('epoch : {} mini-batch : {} loss : {}'.format(epoch,int(count/4), epoch_losses.val))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'epoch_{}.pth'.format(epoch)))


        model.eval()
        epoch_psnr = AverageMeter()
        epoch_sam = AverageMeter()

        Val_train_paths = DataLoader(
        val_paths,
        batch_size=2,
        shuffle=True,
    )

        for paths in Val_train_paths:

            
            # _ 3 63 63
            HR_data = Generate_data(paths)

            train_data = DataLoader(
                HR_data,
                batch_size=VAL_BATCH_SIZE,
                shuffle=False,
                num_workers= 2, 
                pin_memory= True,
                drop_last= True,
            )

            for lr, hr in train_data:

                lr = lr.to(device)
                hr = hr.to(device)
                
                with torch.no_grad():
                    preds = model(
                        lr[:,0,:,:].reshape((VAL_BATCH_SIZE,1,63,63)),
                        lr[:,1,:,:].reshape((VAL_BATCH_SIZE,1,63,63)),
                        lr[:,2,:,:].reshape((VAL_BATCH_SIZE,1,63,63)),
                    )

                    preds = preds.reshape((VAL_BATCH_SIZE,63,63))
                

                epoch_psnr.update(calc_psnr(preds, hr), len(lr))
                epoch_sam.update(SAM_GPU(preds, hr))

        print('epoch : {} average sam is {}'.format(
            epoch,
            epoch_sam.avg)
        )
        print('epoch : {} average psnr is {}'.format(
            epoch,
            epoch_psnr.avg)
        )



        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())       

            print('best psnr epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
            torch.save(best_weights, os.path.join(OUT_DIR, 'best_psnr.pth'))
        
        if epoch_sam.avg < best_sam:
            best_epoch_sam = epoch
            best_sam = epoch_sam.avg
            best_weights_sam = copy.deepcopy(model.state_dict())       

            print('best sam epoch: {}, sam: {:.2f}'.format(best_epoch_sam, best_sam))
            torch.save(best_weights_sam, os.path.join(OUT_DIR, 'best_sam.pth'))


