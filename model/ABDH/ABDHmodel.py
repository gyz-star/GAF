# !/usr/bin/env python
# -*-coding:utf-8 -*-
# import os
import sys
sys.path.append(r"/opt/data/helin/Code/")
from GAF.model.DAH.cbam import ChannelAttention, SpatialAttention
import math

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from GAF.model.DAH import common
from GAF.model.DAH.layer import MultiSpectralAttentionLayer
from GAF.model.DAH.utils import weights_init_normal



class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)


class AttentionNet_jpeg(nn.Module):
    def __init__(self):
        super(AttentionNet_jpeg, self).__init__()
        """
        resnet50的结构：conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
                       1/2              1/2           1/2    1/2    1/2  
        取到layer2止  
        不下采样
        """
        backbone = models.resnet50(pretrained=True)
        model = list(backbone.children())[:6]

        self.feature_extractor = nn.Sequential(*model)
        # print(self.feature_extractor)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(512, 256, 3),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 64, 3),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   nn.Sigmoid())
        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
        self.conv3.apply(weights_init_normal)

        # conv1 = self.feature_extractor[0]
        # conv1.stride = (1, 1)

        layer2_block1 = self.feature_extractor[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        self.fc = nn.Sequential(
            nn.Linear(256, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 256, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # 得到第一个feature_map1
        x = self.feature_extractor(x)
        x = self.conv1(x)
        n, c, h, w = x.shape

        fc_input = torch.sum(x, dim=[2, 3])
        fc_output = self.fc(fc_input).view(n, c, 1, 1)
        channel_attention = fc_output

        # x = self.conv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv2(x)
        x = self.conv3(x)
        return channel_attention, x


class AttentionNet_mutichannel(nn.Module):
    def __init__(self):
        super(AttentionNet_mutichannel, self).__init__()
        """
        resnet50的结构：conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
                       1/2              1/2           1/2    1/2    1/2  
        取到layer2止  
        不下采样
        """
        backbone = models.resnet50(pretrained=True)
        # print(backbone)
        model_1024 = list(backbone.children())[:7]
        model = list(backbone.children())[:6]

        self.feature_extractor = nn.Sequential(*model)
        self.feature_1024 = nn.Sequential(*model_1024)
        # print(self.feature_extractor)

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(512, 256, 3),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True))

        # self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
        #                            nn.Conv2d(256, 64, 3),
        #                            nn.InstanceNorm2d(64),
        #                            nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv2d(64, 1, 1),
        #                            nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 128, 3),
                                   nn.InstanceNorm2d(128),
                                   nn.ReLU(True))

        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(128, 64, 3),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   nn.Sigmoid())

        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
        self.conv3.apply(weights_init_normal)

        # conv1 = self.feature_extractor[0]
        # conv1.stride = (1, 1)

        layer2_block1 = self.feature_extractor[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        layer2_block1 = self.feature_1024[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        # layer3_block1 = self.feature_extractor[6][0]
        # layer3_block1.conv2.stride = (1, 1)
        # layer3_block1.downsample[0].stride = (1, 1)
        #
        # layer4_block1 = self.feature_extractor[6][0]
        # layer4_block1.conv2.stride = (1, 1)
        # layer4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        # 得到第一个feature_map1
        x_1024 = self.feature_1024(x)

        x_512 = self.feature_extractor(x)
        x_256 = self.conv1(F.interpolate(x_512, scale_factor=2, mode='nearest'))
        x_128 = self.conv2(F.interpolate(x_256, scale_factor=2, mode='nearest'))
        x_64 = self.conv3(x_128)
        x_3 = self.conv4(x_64)

        out = (x_1024, x_512, x_256, x_128, x_64, x_3)
        return out


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        """
        resnet50的结构：conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
                       1/2              1/2           1/2    1/2    1/2  
        取到layer2止  
        不下采样
        """
        backbone = models.resnet50(pretrained=True)
        model = list(backbone.children())[:6]
        # in_features = 512
        # out_features = in_features // 2
        # for _ in range(2):
        #     model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2
        # model += [nn.ReflectionPad2d(3),
        #           nn.Conv2d(128, 1, 7),
        #           nn.Sigmoid()]
        # in_features = 512
        # out_features = in_features // 2
        # for _ in range(5):
        #     model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #               nn.ReflectionPad2d(1),
        #               nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=0),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2
        self.feature_extractor = nn.Sequential(*model)
        # print(self.feature_extractor)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(512, 256, 3),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 64, 3),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   nn.Sigmoid())
        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
        self.conv3.apply(weights_init_normal)

        # conv1 = self.feature_extractor[0]
        # conv1.stride = (1, 1)

        layer2_block1 = self.feature_extractor[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        # layer3_block1 = self.feature_extractor[6][0]
        # layer3_block1.conv2.stride = (1, 1)
        # layer3_block1.downsample[0].stride = (1, 1)
        #
        # layer4_block1 = self.feature_extractor[6][0]
        # layer4_block1.conv2.stride = (1, 1)
        # layer4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):  # 得到第一个feature_map1
        x = self.feature_extractor(x)
        x = self.conv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv2(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(in_features)

        self.att = MultiSpectralAttentionLayer(in_features, 14, 14, reduction=16, freq_sel_method='top16')

        # self.conv_jnd = nn.Sequential(
        #     nn.Conv2d(3, 64, 7, 1, 3),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )

        # self.ca_jnd = ChannelAttention(in_features)

        # conv_block = [nn.ReflectionPad2d(1),  # 上上下左右均填充1
        #               nn.Conv2d(in_features, in_features, 3),
        #               # LayerNorm2d(in_features),
        #               nn.InstanceNorm2d(in_features),
        #               nn.ReLU(inplace=True),
        #               nn.ReflectionPad2d(1),
        #               nn.Conv2d(in_features, in_features, 3),
        #               # LayerNorm2d(in_features),
        #               nn.InstanceNorm2d(in_features),
        #               ]

        self.ca = ChannelAttention(in_features)
        self.sa = SpatialAttention()

    def forward(self, x):
        # old
        res = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        # out = self.ca(out) * out
        # 改成：
        out = self.att(out)

        out = self.sa(out) * out

        out = out + res
        out = self.relu(out)

        # res = x
        # out = self.conv1(x)
        # out = self.in1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.in2(out)
        #
        # if jnd is not None:
        #     j_out = self.ca_jnd(jnd)
        #     out = j_out * out
        #
        # # out = self.att(out)
        #
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        # # out = channel_a * out
        # # out = spatial_a * out
        #
        # out = out + res
        # out = self.relu(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)  # torch.sigmoid(
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # [b,1]


class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=4, n_residual_blocks=9):
        super(Encoder, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        downsample = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(int(math.log(dct_kernel, 2))):
            downsample += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.downsample = nn.Sequential(*downsample)

        # Residual blocks
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, output_nc, 3),  # mean
            # nn.ReLU()  # max_min
            nn.Sigmoid()
        )

    def forward(self, x):
        """old forward"""
        x = self.conv1(x)
        x = self.downsample(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x

        # y = self.conv1_jpeg(jpeg)
        #
        # x = self.conv1(x)
        # x = self.downsample(x)
        # res = x
        #
        # x = x + y
        # for i in range(self.n_residual_blocks - 1):
        #     x = self.body[i](x) + y
        # x = self.body[8](x)
        #
        # x = x + res
        # x = self.tail(x)
        # return x


class Encoder2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=4, n_residual_blocks=9):
        super(Encoder2, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        # self.conv_jnd = nn.Sequential(
        #     nn.Conv2d(3, 64, 7, 1, 3),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        downsample = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(int(math.log(dct_kernel, 2))):
            downsample += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.downsample = nn.Sequential(*downsample)

        # Residual blocks
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, output_nc, 3),  # mean
            # nn.ReLU()  # max_min
            nn.Sigmoid()
        )

    def forward(self, x, jnd=None, channel_a=1, spatial_a=1):
        """old forward"""
        x = self.conv1(x)
        x = self.downsample(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x
        # if jnd != None:
        #     jnd = self.conv_jnd(jnd)
        # # j_out = self.ca(j_out)
        #
        # x = self.conv1(x)
        # x = self.downsample(x)
        # res = x
        #
        # for i in range(self.n_residual_blocks):
        #     x = self.body[i](x, jnd, channel_a, spatial_a)
        #
        #
        # x = x + res
        # x = self.tail(x)
        # return x


class Decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9, dct_kernel=8):
        super(Decoder, self).__init__()

        self.upconv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 256, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Initial convolution block
        model = []
        # model = [  # LayerNorm2d(192),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(input_nc, 512, 7),
        #     # nn.GroupNorm(8, 64),
        #     # LayerNorm2d(64),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True)]

        in_features = 256

        # Upsampling
        # out_features = in_features // 2
        # for _ in range(3):
        #     model += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
        #               # nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Output layer
        tail = [nn.ReflectionPad2d(1),
                nn.Conv2d(64, output_nc, 3),  # mean
                # nn.ReLU()  # max_min
                nn.Sigmoid()
                # nn.PReLU(1)
                ]

        self.model = nn.Sequential(*model)
        self.tail = nn.Sequential(*tail)

        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        x = self.upconv1(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.upconv2(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.upconv3(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.tail(x)
        return x


class Decoder_cbam(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=4, n_residual_blocks=9):
        super(Decoder_cbam, self).__init__()

        self.n_residual_blocks = n_residual_blocks
        tmp_channel = 128

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, tmp_channel, 7),
            nn.InstanceNorm2d(tmp_channel),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        in_features = tmp_channel
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))

        conv = common.default_conv

        tail = [
            nn.Conv2d(tmp_channel, 64, 1, padding=0, stride=1),
            # conv(64, 64, 3),
            # common.Upsampler(conv, scale=dct_kernel, n_feats=64, act=False),
            conv(64, output_nc, 3),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None, dilation=1):
        super(ResBlock, self).__init__()
        feature = 64
        channel_out = channel_in if channel_out is None else channel_out
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, feature, kernel_size=3, padding=1 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.InstanceNorm2d(feature, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Conv2d((feature + channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)
        residual = self.conv1_1(residual)
        residual = self.conv2_1(residual)
        residual = self.conv3_1(residual)
        residual = self.conv4_1(residual)

        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)

        return out


class Decoder_res(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=3):
        super(Decoder_res, self).__init__()

        self.n_residual_blocks = n_residual_blocks
        tmp_channel = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=tmp_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(tmp_channel),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        in_features = tmp_channel
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResBlock(in_features))

        tail = [
            nn.Conv2d(in_channels=tmp_channel, out_channels=output_nc, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x


import torch
import torch.nn as nn

# --------------------------------#
# 从torch官方可以下载resnet50的权重
# --------------------------------#
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# -----------------------------------------------#
# 此处为定义3*3的卷积，即为指此次卷积的卷积核的大小为3*3
# -----------------------------------------------#
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# -----------------------------------------------#
# 此处为定义1*1的卷积，即为指此次卷积的卷积核的大小为1*1
# -----------------------------------------------#
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ----------------------------------#
# 此为resnet50中标准残差结构的定义
# conv3x3以及conv1x1均在该结构中被定义
# ----------------------------------#
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        # --------------------------------------------#
        # 当不指定正则化操作时将会默认进行二维的数据归一化操作
        # --------------------------------------------#
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ---------------------------------------------------#
        # 根据input的planes确定width,width的值为
        # 卷积输出通道以及BatchNorm2d的数值
        # 因为在接下来resnet结构构建的过程中给到的planes的数值不相同
        # ---------------------------------------------------#
        width = int(planes * (base_width / 64.)) * groups
        # -----------------------------------------------#
        # 当步长的值不为1时,self.conv2 and self.downsample
        # 的作用均为对输入进行下采样操作
        # 下面为定义了一系列操作,包括卷积，数据归一化以及relu等
        # -----------------------------------------------#
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # --------------------------------------#
    # 定义resnet50中的标准残差结构的前向传播函数
    # --------------------------------------#
    def forward(self, x):
        identity = x
        # -------------------------------------------------------------------------#
        # conv1*1->bn1->relu 先进行一次1*1的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv3*3->bn2->relu 先进行一次3*3的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv1*1->bn3 先进行一次1*1的卷积之后进行数据归一化操作
        # -------------------------------------------------------------------------#
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # -----------------------------#
        # 若有下采样操作则进行一次下采样操作
        # -----------------------------#
        if self.downsample is not None:
            identity = self.downsample(identity)
        # ---------------------------------------------#
        # 首先是将两部分进行add操作,最后过relu来增加非线性因素
        # concat（堆叠）可以看作是通道数的增加
        # add（相加）可以看作是特征图相加，通道数不变
        # add可以看作特殊的concat,并且其计算量相对较小
        # ---------------------------------------------#
        out += identity
        out = self.relu(out)

        return out


# --------------------------------#
# 此为resnet50网络的定义
# input的大小为224*224
# 初始化函数中的block即为上面定义的
# 标准残差结构--Bottleneck
# --------------------------------#
class ResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        # ---------------------------------------------------------#
        # 使用膨胀率来替代stride,若replace_stride_with_dilation为none
        # 则这个列表中的三个值均为False
        # ---------------------------------------------------------#
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        # ----------------------------------------------#
        # 若replace_stride_with_dilation这个列表的长度不为3
        # 则会有ValueError
        # ----------------------------------------------#
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group
        # -----------------------------------#
        # conv1*1->bn1->relu
        # 224,224,3 -> 112,112,64
        # -----------------------------------#
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # ------------------------------------#
        # 最大池化只会改变特征图像的高度以及
        # 宽度,其通道数并不会发生改变
        # 112,112,64 -> 56,56,64
        # ------------------------------------#
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64   -> 56,56,256
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)

        # 56,56,256  -> 28,28,512
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # 28,28,512  -> 14,14,1024
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # --------------------------------------------#
        # 自适应的二维平均池化操作,特征图像的高和宽的值均变为1
        # 并且特征图像的通道数将不会发生改变
        # 7,7,2048 -> 1,1,2048
        # --------------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ----------------------------------------#
        # 将目前的特征通道数变成所要求的特征通道数（1000）
        # 2048 -> num_classes
        # ----------------------------------------#
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # -------------------------------#
        # 部分权重的初始化操作
        # -------------------------------#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # -------------------------------#
        # 部分权重的初始化操作
        # -------------------------------#
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # --------------------------------------#
    # _make_layer这个函数的定义其可以在类的
    # 初始化函数中被调用
    # block即为上面定义的标准残差结构--Bottleneck
    # --------------------------------------#
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # -----------------------------------#
        # 在函数的定义中dilate的值为False
        # 所以说下面的语句将直接跳过
        # -----------------------------------#
        if dilate:
            self.dilation *= stride
            stride = 1
        # -----------------------------------------------------------#
        # 如果stride！=1或者self.inplanes != planes * block.expansion
        # 则downsample将有一次1*1的conv以及一次BatchNorm2d
        # -----------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # -----------------------------------------------#
        # 首先定义一个layers,其为一个列表
        # 卷积块的定义,每一个卷积块可以理解为一个Bottleneck的使用
        # -----------------------------------------------#
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # ------------------------------#
    # resnet50的前向传播函数
    # ------------------------------#
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_64 = self.relu(x)
        # x = self.maxpool(x)

        x_128 = self.layer1(x_64)
        x_256 = self.layer2(x_128)
        x_512 = self.layer3(x_256)
        x_1024 = self.layer4(x_512)
        x = self.avgpool(x_1024)
        # --------------------------------------#
        # 按照x的第1个维度拼接（按照列来拼接，横向拼接）
        # 拼接之后,张量的shape为(batch_size,2048)
        # --------------------------------------#
        # x = torch.flatten(x, 1)
        # --------------------------------------#
        # 过全连接层来调整特征通道数
        # (batch_size,2048)->(batch_size,1000)
        # --------------------------------------#
        # x = self.fc(x)
        return [x_64, x_128, x_256, x_512, x_1024]


# F = torch.randn(16, 3, 224, 224)
# print("As begin,shape:", format(F.shape))
# resnet = ResNet(Bottleneck, [3, 4, 6, 3])
# F = resnet(F)
# print(F.shape)


if __name__ == '__main__':
    # from torchviz import make_dot
    # from torchsummary import summary

    device = torch.device("cpu")

    dct = DCT(device, N=4)

    att = AttentionNet()
    att_jpeg = ResNet()

    # print(att)
    # gen2 = GeneratorImage(input_nc=3, output_nc=3)
    disc = Discriminator(input_nc=192)
    tmp = torch.arange(2 * 3 * 32 * 32)

    input2 = torch.rand(1, 3, 256, 256)
    jpeg = torch.rand(1, 3, 256, 256)
    jnd = torch.rand(1, 3, 8, 8)
    dct_ = dct(input2)
    # a, b, c, d, e, f = att(input2)
    ca, sa = att_jpeg(input2)

    #
    # encoder = Encoder2(input_nc=3, output_nc=48, dct_kernel=4)
    # decoder = Decoder_res(input_nc=3, output_nc=3)
    # output1 = encoder(input2, None, ca, sa)
    # output2 = decoder(input2)
    # print(decoder)
    # print(output1.shape, output2.shape)

    # summary(encoder,(3, 256, 256))
    # summary(decoder,(192, 32, 32))
    # backbone = models.resnet50(pretrained=True)

    # print(backbone)
    # g = make_dot(out)
    # g.render('espnet_model2', view=False)
