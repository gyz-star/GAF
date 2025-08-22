import torch
import torch.nn as nn
import torch.nn.functional as F
import myUtils


class DeepHomographyModel(nn.Module):
    def __init__(self):
        super(DeepHomographyModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 64, 3, padding=1)  # 输出1：【b 64 256 256】
        self.conv_layer2 = nn.Conv2d(64, 64, 3, padding=1)  # 输出2：【b 64 256 256】输出3：经一个maxpool-》【b 64 128 128】
        self.conv_layer3 = nn.Conv2d(64, 64, 3, padding=1)  # 输出4：【b 64 128 128】
        self.conv_layer4 = nn.Conv2d(64, 64, 3, padding=1)  # 输出5：【b 64 128 128】输出6：经一个maxpool-》【b 64 64 64】
        self.conv_layer5 = nn.Conv2d(64, 128, 3, padding=1)  # 输出7：【b 128 64 64】
        self.conv_layer6 = nn.Conv2d(128, 128, 3, padding=1)  # 输出8：【b 128 64 64】 输出9：经一个maxpool-》【b 128 32 32】
        self.conv_layer7 = nn.Conv2d(128, 128, 3, padding=1)  # 输出10：【b 128 32 32】
        self.conv_layer8 = nn.Conv2d(128, 128, 3, padding=1)  # 输出11：【b 128 32 32】
        self.fc_layer1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc_layer2 = nn.Linear(1024, 9)
        self.fc_layer2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.batch_norm8 = nn.BatchNorm2d(128)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)

        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer5(out)
        out = self.batch_norm5(out)
        out = F.relu(out)

        out = self.conv_layer6(out)
        out = self.batch_norm6(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer7(out)
        out = self.batch_norm7(out)
        out = F.relu(out)

        out = self.conv_layer8(out)
        out = self.batch_norm8(out)
        out = F.relu(out)
        out = out.view(-1, 128 * 16 * 16)

        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        out = out.reshape(batch_size, 3, 3)
        return out


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # 定义本地化网络，用于估计空间变换的参数
        self.localization = nn.Sequential(
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
        # 定义空间变换网络，用于预测空间变换的参数
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 3)
        )
        # 初始化空间变换网络的权重和偏置
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

    def forward(self, x):
        # 使用本地化网络对输入图像进行特征提取
        xs = self.localization(x)
        # 将特征张量展开成一维张量
        xs = xs.view(-1, 512)
        # 使用空间变换网络预测空间变换的参数
        theta = self.fc_loc(xs)
        # 将一维张量转换成二维张量，用于执行仿射变换
        Homography = theta.view(-1, 3, 3)
        return Homography


class STN2(nn.Module):
    def __init__(self):
        super(STN2, self).__init__()
        # 定义本地化网络，用于估计空间变换的参数
        self.ConvNet = nn.Sequential(
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
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, 8)
        # 初始化空间变换网络的权重和偏置
        # self.localization_fc2.weight.data.fill_(0)
        # 修改全连接层2的偏移量
        # self.localization_fc2.bias.data.copy_(torch.tensor([0, 0, 0, 255, 255, 0, 255, 255], dtype=torch.float))

    def forward(self, x):
        batch_size = x.size(0)
        # 使用本地化网络对输入图像进行特征提取
        features = self.ConvNet(x).view(batch_size, -1)
        # 使用特征计算基准点集合
        features = self.localization_fc1(features)
        # print(features)
        points = self.localization_fc2(features).view(batch_size, 4, 2)
        return points


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
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

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 512)
        # print('到了这里1')
        theta = self.fc_loc(xs)
        # print('到了这里2')
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def forward(self, x):
        # transform the input
        x, theta = self.stn(x)
        return x, theta

