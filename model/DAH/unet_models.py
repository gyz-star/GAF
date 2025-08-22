""" Full assembly of the parts to form the complete network """

import sys
sys.path.append(r"/opt/data/helin/Code/")
from GAF.model.ABDH.ABDHmodel import AttentionNet, ResNet
from GAF.model.DAH.unet_parts import *



class UNet_old(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):
        super(UNet_old, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        self.S_outc = (OutCv(64, out_channels))
        # self.S_outc = (AddOutCv(64, out_channels)) # 下采样缩小尺寸

    def forward(self, y):
        y1 = self.S_inc(y)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2 = self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        x_out = self.S_outc(y_up4)
        # logits = self.C_outc(x)

        return x_out


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.C_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.C_down1 = (Down(64, 128, norm_layer))
        self.C_down2 = (Down(128, 256, norm_layer))
        self.C_down3 = (Down(256, 512, norm_layer))
        self.C_down4 = (Down(512, 1024 // factor, norm_layer))
        self.C_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.C_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.C_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.C_up4 = (Up(128, 64, bilinear, norm_layer))
        self.C_add = (AddOutCv(64, out_channels))  # k=4 使用AddOutCv  k=2 使用AddOutCv_kernel2
        # self.C_outc = (OutConv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, x, y):
        y1 = self.S_inc(y)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2 = self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        x1 = self.C_inc(x)
        x2 = self.C_down1(x1)
        x3 = self.C_down2(x2)
        x4 = self.C_down3(x3)
        x5 = self.C_down4(x4)
        x = self.C_up1(x5, x4, None)
        x = self.C_up2(x, x3, y_up1)
        x = self.C_up3(x, x2, y_up2)
        x = self.C_up4(x, x1, y_up3)
        x_out = self.C_add(x, y_up4)
        # logits = self.C_outc(x)

        return x_out


class UNet_FA_add_att(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):
        super(UNet_FA_add_att, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.C_inc = (DoubleCv(n_channels*3, 64, norm_layer=norm_layer))
        self.C_down1 = (Down(64, 128, norm_layer))
        self.C_down2 = (Down(128, 256, norm_layer))
        self.C_down3 = (Down(256, 512, norm_layer))
        self.C_down4 = (Down(512, 1024 // factor, norm_layer))
        self.C_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.C_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.C_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.C_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.C_add = (AddOutCv(64, out_channels))  # k=4 使用AddOutCv  k=2 使用AddOutCv_kernel2
        self.C_outc = (OutCv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))  # 5.23改了，用att cat在一起
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, secret, other):
        y1 = self.S_inc(secret)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2 = self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        # x = x + attention
        x1 = self.C_inc(other)
        x2 = self.C_down1(x1)
        x3 = self.C_down2(x2)
        x4 = self.C_down3(x3)
        x5 = self.C_down4(x4)
        other = self.C_up1(x5, x4, None)
        other = self.C_up2(other, x3, y_up1)
        other = self.C_up3(other, x2, y_up2)
        other = self.C_up4(other, x1, y_up3)
        # x_out = self.C_add(x, y_up4)
        x_out = self.C_outc(other + y_up4)

        return x_out



class UNet_FA_add(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):
        super(UNet_FA_add, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.C_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.C_down1 = (Down(64, 128, norm_layer))
        self.C_down2 = (Down(128, 256, norm_layer))
        self.C_down3 = (Down(256, 512, norm_layer))
        self.C_down4 = (Down(512, 1024 // factor, norm_layer))
        self.C_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.C_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.C_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.C_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.C_add = (AddOutCv(64, out_channels))  # k=4 使用AddOutCv  k=2 使用AddOutCv_kernel2
        self.C_outc = (OutCv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))  # 5.23改了，用att cat在一起
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, secret, other):
        y1 = self.S_inc(secret)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2 = self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        # x = x + attention
        x1 = self.C_inc(other)
        x2 = self.C_down1(x1)
        x3 = self.C_down2(x2)
        x4 = self.C_down3(x3)
        x5 = self.C_down4(x4)
        other = self.C_up1(x5, x4, None)
        other = self.C_up2(other, x3, y_up1)
        other = self.C_up3(other, x2, y_up2)
        other = self.C_up4(other, x1, y_up3)
        # x_out = self.C_add(x, y_up4)
        x_out = self.C_outc(other + y_up4)

        return x_out


class UNet_FA_add_best(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):#n_channels=3, out_channels=3
        super(UNet_FA_add_best, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.C_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))#3 256 256 -》64 256 256
        self.C_down1 = (Down(64, 128, norm_layer))#64 256 256 -》128 128 128
        self.C_down2 = (Down(128, 256, norm_layer))#128 128 128 -》256 64 64
        self.C_down3 = (Down(256, 512, norm_layer))#256 64 64 -》512 32 32
        self.C_down4 = (Down(512, 1024 // factor, norm_layer))#512 32 32 -》1024 16 16
        self.C_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.C_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.C_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.C_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.C_add = (AddOutCv(64, out_channels))  # k=4 使用AddOutCv  k=2 使用AddOutCv_kernel2
        self.C_add = (OutCv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, x, y, attention):#x为secret y为cover
        y1 = self.S_inc(y)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2 = self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        x = x + attention
        x1 = self.C_inc(x)
        x2 = self.C_down1(x1)
        x3 = self.C_down2(x2)
        x4 = self.C_down3(x3)
        x5 = self.C_down4(x4)
        x = self.C_up1(x5, x4, None)
        x = self.C_up2(x, x3, y_up1)
        x = self.C_up3(x, x2, y_up2)
        x = self.C_up4(x, x1, y_up3)
        x_out = self.C_add(x + y_up4)
        # logits = self.C_outc(x)

        return x_out

class UNet_FA_add_best_32(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):#n_channels=3, out_channels=3
        super(UNet_FA_add_best_32, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1#bilinear=false factor=1

        self.C_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))#3 32 32 -》64 32 32
        self.C_down1 = (Down(64, 128 // factor, norm_layer))#64 32 32 -》128 16 16
        self.C_up1 = (Up(128, 64, bilinear, norm_layer))
        self.C_add = (OutCv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))#3 32 32 -》64 32 32
        self.S_down1 = (Down(64, 128, norm_layer))#64 32 32 -》128 16 16
        self.S_up1 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, x, y, attention):#x为secret y为cover
        y1 = self.S_inc(y)
        y2 = self.S_down1(y1)
        y_up1 = self.S_up1(y2, y1, None)
        # logits = self.S_outc(y_up4)

        x = x + attention
        x1 = self.C_inc(x)
        x2 = self.C_down1(x1)
        x = self.C_up1(x2, x1, None)
        x_out = self.C_add(x + y_up1)
        # logits = self.C_outc(x)

        return x_out

if __name__ == "__main__":
    # from torchviz import make_dot
    # from torchsummary import summary
    # device = torch.device("cuda:3")

    Hnet = UNet_FA_add_best_32(3, 3, bilinear=False, norm_layer=nn.InstanceNorm2d)
    print(Hnet)
    attention = AttentionNet()
    attention2 = ResNet()
    # print(Hnet)
    inputs = torch.randn(2, 3, 32, 32)
    # attlist = attention(inputs)
    attlist = attention2(inputs)

    att = torch.randn(2, 1, 32, 32)
    outputs = Hnet(inputs, inputs, 0)
    print(outputs.shape)
