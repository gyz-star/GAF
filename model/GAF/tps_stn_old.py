import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import kornia

sys.path.append(r"/opt/data/helin/code/")
from GAF.model.GAF.tps_grid_gen import TPSGridGen
from GAF.model.GAF.DCN import DeformConv2D
from GAF.myUtils import *

class DCN(nn.Module):
    def __init__(self, num_output):
        super(DCN, self).__init__()
        self.COV = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),  # [N, 64, H, W]
            nn.MaxPool2d(2, 2),  # [N, 64, H/2, W/2]
            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),  # [N, 128, H/2, W/2]
            nn.MaxPool2d(2, 2),  # [N, 128, H/4, W/4]
        )
        self.offsets1 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.DCN1 = DeformConv2D(128, 256, kernel_size=3, padding=1)
        self.DCN1_bn = nn.BatchNorm2d(256)
        self.DCN1_re = nn.ReLU(True)  # [N, 256, H/4, W/4]
        self.DCN1_maxpool = nn.MaxPool2d(2, 2)  # [N, 256, H/8, W/8]

        self.offsets2 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.DCN2 = DeformConv2D(256, 512, kernel_size=3, padding=1)
        self.DCN2_bn = nn.BatchNorm2d(512)  # [N, 512, H/8, W/8]
        self.DCN2_re = nn.ReLU(True)
        self.DCN2_maxpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_output)

    def forward(self, x):
        x = self.COV(x)

        offsets1 = self.offsets1(x)
        x = self.DCN1(x, offsets1)
        x = self.DCN1_bn(x)
        x = self.DCN1_re(x)
        x = self.DCN1_maxpool(x)

        offsets2 = self.offsets2(x)
        x = self.DCN2(x, offsets2)
        x = self.DCN2_bn(x)
        x = self.DCN2_re(x)
        x = self.DCN2_maxpool(x)

        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.COV = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),  # [N, 64, H, W]
            nn.MaxPool2d(2, 2),  # [N, 64, H/2, W/2]
            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),  # [N, 128, H/2, W/2]
            nn.MaxPool2d(2, 2),  # [N, 128, H/4, W/4]
            nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),  # [N, 256, H/4, W/4]
            nn.MaxPool2d(2, 2),  # [N, 256, H/8, W/8]
            nn.Conv2d(256, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),  # [N, 512, H/8, W/8]
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_output)

    def forward(self, x):
        x = self.COV(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, arg):
        super(BoundedGridLocNet, self).__init__()
        if arg.IsDCN == 0:
            self.cnn = CNN(grid_height * grid_width * 2)
        else:
            self.cnn = DCN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points, arg):
        super(UnBoundedGridLocNet, self).__init__()
        if arg.IsDCN == 0:
            self.cnn = CNN(grid_height * grid_width * 2)
        else:
            self.cnn = DCN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        # pdb.set_trace()
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x



class STNTpsNet(nn.Module):

    def __init__(self, args,device):
        super(STNTpsNet, self).__init__()
        self.args = args
        self.device=device
        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }['bounded_stn']
        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points, args)
        self.Gaussian=GaussianBlurConv(3)

        self.tps = TPSGridGen(args.image_height//8, args.image_width//8,
                              target_control_points)
    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.args.image_height//8, self.args.image_width//8, 2)
        return grid




class estimateSigma(nn.Module):
    def __init__(self,device):
        super(estimateSigma, self).__init__()
        self.device=device
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),  # b 16 128 128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # b 16 64 64
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # b 32 32 32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # b 32 16 16
        )
        self.fc = nn.Sequential(
            torch.nn.Linear(32 * 16 * 16, 16 * 16),
            torch.nn.Linear(16 * 16, 64),
            torch.nn.Sigmoid()
        )
        #线性层初始化
        # weight=torch.Tensor(list( np.arange(0, 1+0.001, 1 / 63)))
        # bias = weight.view(-1)
        # self.fc[1].bias.data.copy_(bias)
        # self.fc[1].weight.data.zero_()


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)# batch 64
        return x
