#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   复现论文使用的神经网络 
@File    :   net.py
@Time    :   2020/11/26 17:36:29
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/11/26 17:36:29
'''

import torch.nn as nn


class SRCNN(nn.Module):

    def __init__(self, in_channles=1):
        super(SRCNN,self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channles, 64, 9, 1, 9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channles, 5, 1, 5 // 2)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(in_channles, 64, 9, 1, 9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channles, 5, 1, 5//2)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(in_channles, 64, 9, 1, 9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channles, 5, 1, 5//2)
        )


    def forward(self,x1, x2, x3):
        y1 = self.model1(x1)
        y2 = self.model2(x2)
        y3 = self.model3(x3)

        y = (y1 + y2 + y3) / 2

        return y