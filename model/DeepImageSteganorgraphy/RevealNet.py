# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: Reveal.py
@time: 2018/3/20

"""

import torch.nn as nn
import torch.nn.functional as F

class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            # 下采样三次
            nn.Conv2d(nc, nhf, 3, 1, 1), # 0
            nn.BatchNorm2d(nhf),# 1
            nn.ReLU(True),# 2
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),# 3
            nn.BatchNorm2d(nhf * 2),# 4
            nn.ReLU(True),# 5
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),# 6
            nn.BatchNorm2d(nhf * 4),# 7
            nn.ReLU(True),# 8
            # 上采样三次
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),# 9
            nn.BatchNorm2d(nhf * 2),# 10
            nn.ReLU(True),# 11
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),# 12
            nn.BatchNorm2d(nhf),# 13
            nn.ReLU(True),# 14
            nn.Conv2d(nhf, nc, 3, 1, 1),# 15
            output_function()# 16
        )

    def forward(self, output):
        # print('==================第六层================')
        # show1=self.main[0].state_dict()['weight']
        # print(show1)
        # print('==================第9层================')
        # show2=self.main[15].state_dict()['weight']
        # print(show2)
        # with open("/opt/data/helin/Code/new_stn/hist/our/ceng0.txt", "w") as file:
        #     for i in range(64):
        #         for j in range(3):
        #             for x in range(3):
        #                 for y in range(3):
        #                     file.write(str(show1[i][j][x][y].item()))
        #                     file.write(',')
        #             file.write('\n')
        #             file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        # with open("/opt/data/helin/Code/new_stn/hist/our/ceng15.txt", "w") as file:
        #     for i in range(3):
        #         for j in range(64):
        #             for x in range(3):
        #                 for y in range(3):
        #                     file.write(str(show2[i][j][x][y].item()))
        #                     file.write(',')
        #             file.write('\n')
        #             file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        #         file.write('\n')
        output=self.main(output)
        return output
