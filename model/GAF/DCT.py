import math

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DCT(nn.Module):
    def __init__(self, device, N=8, in_channal=3):
        super(DCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N
        self.in_channal = in_channal
        self.out_channal = N * N * in_channal
        self.device = device

        # rgb to ycbcr
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        # ycbcr to rgb
        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix
        self.reYcbcr.weight.requires_grad = False

        # 3 H W -> 3*N*N  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.rearrange(N=N, zigzag=True)).float().unsqueeze(1)
        self.dct_conv.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)  # 64 1 8 8
        self.dct_conv.weight.requires_grad = False

        self.idct_conv = nn.ConvTranspose2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)
        self.idct_conv.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)
        self.idct_conv.weight.requires_grad = False

    def norm(self, x, eps=1e-5):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
        b, c, _, _ = x.shape
        data_max, data_min = torch.zeros(b, c, requires_grad=False).to(self.device), \
                             torch.zeros(b, c, requires_grad=False).to(self.device)

        for i in range(b):
            for j in range(c):
                channel = x[i][j]
                max_channel, min_channel = torch.max(channel), torch.min(channel)
                data_max[i][j], data_min[i][j] = max_channel, min_channel

        x = x - data_min[..., None, None]
        x = x / ((data_max - data_min)[..., None, None] + eps)

        return data_max, data_min, x

    def renorm(self, x, data_max, data_min, eps=1e-5):
        b, c, _, _ = x.shape
        x = x * ((data_max - data_min)[..., None, None] + eps)
        x = x + data_min[..., None, None]
        # x.requires_grad = False
        return x

    def forward(self, x):
        '''
        x:  B C H W, 0-1. RGB
        YCbCr:  b c h w, YCBCR
        DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        '''
        ycbcr = self.Ycbcr(x)
        dct = self.dct_conv(ycbcr)
        # dct = self.output_channel_3(dct)

        return dct

    def output_channel_3(self, img):
        c = self.N ** 2
        image_conv_channels = []
        for i in range(3):
            imgg = img[:, i * c:(i + 1) * c, :, :]
            image_conv = imgg.permute(0, 2, 3, 1)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], self.N, self.N)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                      image_conv.shape[1] * image_conv.shape[2],
                                                      image_conv.shape[3] * image_conv.shape[4])

            image_conv.unsqueeze_(1)
            image_conv_channels.append(image_conv)
        image_conv_stacked = torch.cat(image_conv_channels, dim=1)
        return image_conv_stacked

    def output_channel_N(self, img):
        image_conv_channels = []
        for i in range(3):
            image_conv = img[:, i:(i + 1), :, :]
            image_conv = image_conv.squeeze(1)
            image_conv = image_conv.view(image_conv.shape[0], int(image_conv.shape[1] / self.N), self.N,
                                         int(image_conv.shape[2] / self.N), self.N)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2],
                                                      image_conv.shape[4] * image_conv.shape[3])
            image_conv = image_conv.permute(0, 3, 1, 2)
            image_conv.squeeze_(1)
            image_conv_channels.append(image_conv)
        image_conv_stacked = torch.cat(image_conv_channels, dim=1)
        return image_conv_stacked

    def reverse(self, x):
        # x = self.output_channel_N(x)
        dct = self.idct_conv(x)
        rgb = self.reYcbcr(dct)
        return rgb

    def rearrange(self, N=8, zigzag=True):
        dct_weight = np.zeros((N * N, N, N))
        for k in range(N * N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        out_weight = dct_weight
        if zigzag:
            out_weight = self.get_zigzag_order(dct_weight, N=N)  # from low frequency to high frequency
        return out_weight  # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_zigzag_order(self, src_weight, N=8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy()  # (N*N) * N * N
        # index = [0, 1, 3, 6, 4, 2, 5, 7, 8]
        for k in range(array_size - 1):
            if (i == 0 or i == N - 1) and j % 2 == 0:
                j += 1
            elif (j == 0 or j == N - 1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k + 1, ...] = src_weight[index, ...]
        return rearrange_weigth


def calc_channel_all_var(data, mean, size=192):  # 计算标准差的辅助函数
    data, mean = np.array(data), np.array(mean)
    data = np.transpose(data, [1, 2, 0])
    channel_var = np.sum((data - mean) ** 2, axis=(0, 1))
    return channel_var


def calc_channel_mean(x):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    b, c, _, _ = x.shape
    mean = torch.mean(x, dim=(2,3))
    var = torch.var(x, dim=(2,3))

    data = x
    x = x - mean[..., None, None]
    x = x / var[..., None, None]

    return mean, var, x



if __name__ == '__main__':
    # from torchviz import make_dot
    # from torchsummary import summary

    device = torch.device("cpu")

    dct = DCT(device=device, N=4)

    # test
    # path = '/opt/data/xiaobin/ABDH_UDH/ABDH/result_img/realimg.png'
    # path = r'C:\Users\Dannis\remote\ABDH\result_img\realimg.png'
    # img = cv2.imread(path)
    # img = cv2.resize(img,[128,128])
    # img = torch.from_numpy(img / 255).unsqueeze(0).permute(0, 3, 1, 2).float()
    img = torch.rand(5,3,128,128)
    img_dct = dct(img)

    # x, mean, var = calc_channel_mean(img_dct)
    # x1 = np.array(img_dct)
    # x2 = np.array(x)

    dct2 = dct.reverse(img_dct)
    print("1")
