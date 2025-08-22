import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision



# 输入一张图片返回高频和低频
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


# 分割图像为k x k块
def split_image_into_blocks(image, k):
    array1d = torch.linspace(start=0, end=image.shape[0], steps=k)  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点
    array1d2 = torch.linspace(start=0, end=image.shape[1], steps=k)  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点
    x, y = torch.meshgrid(array1d, array1d2)  # 生成网格
    identity_grid = torch.stack((y, x), 2)[None, ...]
    blocks = []
    for i in range(0, k - 1):
        for j in range(0, k - 1):
            block = image[int(identity_grid[0][i][j][0]):int(identity_grid[0][i][j + 1][0]),
                    int(identity_grid[0][i][j][1]):int(identity_grid[0][i + 1][j][1]), :]
            top_left = identity_grid[0][i][j]
            top_right = identity_grid[0][i][j + 1]
            bottom_left = identity_grid[0][i + 1][j]
            bottom_right = identity_grid[0][i + 1][j + 1]
            gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            blocks.append((block, top_left, top_right, bottom_left, bottom_right, variance))
    return blocks


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss


def getInitPoints():
    X = torch.tensor([0, 0, 255, 255], dtype=torch.float32).resize(4, 1)
    Y = torch.tensor([0, 255, 0, 255], dtype=torch.float32).resize(4, 1)
    initial_bias = torch.cat([X, Y], dim=1)
    return initial_bias


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH
    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

