# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
sys.path.append(r"/opt/data/helin/Code/")
from GAF.model.ABDH.utils import weights_init_normal


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

        conv_block = [nn.ReflectionPad2d(1),  # 上上下左右均填充1
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorImage(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(GeneratorImage, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class GeneratorImage32(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=1):
        super(GeneratorImage32, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(1):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),#H->1/2H
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(1):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


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


if __name__ == '__main__':
    # net = Discriminator(3)
    gen1 = AttentionNet()
    print(gen1)
    gen2 = GeneratorImage()
    print(gen2)
    disc = Discriminator()
    print(disc)
    img = torch.randn(1, 3, 256, 256)
    sec = torch.randn(2, 3, 512, 512)
    # out = net(img)
    # print(out.shape)
    attention_mask = gen1(img)
    attention_mask[0 <= attention_mask and attention_mask <= 1] = 1.0
    # print(attention_mask.shape)
    # summary(gen1.cuda(), [(3, 512, 512)])
    extract_img = gen2(target_img)
    disc_out = disc(img)
    # print(disc_out.shape)
    # print(attention_mask.shape)
    # print(attention_mask.shape, target_img.shape, extract_img.shape, disc_out.shape)
    # print(out)
    # backbone = models.resnet50(pretrained=True)
    #
    # print(backbone)
