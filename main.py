import argparse
import os
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from PIL import Image
import sys

sys.path.append(r"/opt/data/helin/Code/")
from GAF.data.dataset import MyImageFolder
import GAF.transformed as transforms
from GAF.model.GAF.tps_stn_old import STNTpsNet
from GAF.model.GAF.tps_stn_old import estimateSigma
# DS的encoder和decoder
from GAF.model.DeepImageSteganorgraphy.HidingUNet import UnetGenerator
from GAF.model.DeepImageSteganorgraphy.RevealNet import RevealNet
from GAF.Train.DS.TrainDS import Train as Train_ds
from GAF.Train.DAH.TrainDAH import Train as Train_dah
from GAF.Train.ABDH.TrainABDH import Train as Train_abdh
# DAH的encoder和decoder
from GAF.model.DAH.unet_models import UNet_FA_add_best
from GAF.model.DAH.RevealNet import RevealNet as RevealNet_DAH

from GAF.myUtils import VGGLoss
# ABDH的模型
import GAF.model.ABDH.model as ABDH


# python main_ds.py --gpu_id 0 --grid_size 8 --IsDCN 0 --save_models_path /opt/data/helin/Code/new_stn/checkpoints/important/NODCN_8/ --logdir /opt/data/helin/Code/new_stn/log/important/NODCN_8/
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs to train', default=10, type=int)
    parser.add_argument('--lr', help='learning rate', default='0.001', type=float)
    parser.add_argument('--batch_size', help='batch size', default='8', type=int)
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--stego', type=str, default='DS')
    parser.add_argument('--filename', type=str, default='log_train.txt')
    parser.add_argument('--Hnet', type=str,
                        default='/opt/data/helin/Code/new_stn/checkpoints/DeepImageSteganorgraphy/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth')
    parser.add_argument('----Rnet', type=str,
                        default='/opt/data/helin/Code/new_stn/checkpoints/DeepImageSteganorgraphy/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth')
    parser.add_argument('--cp_path', type=str,
                        default='/opt/data/helin/Code/new_stn/test_image/test_DAH/199.pth')
    parser.add_argument('--span_range', type=int, default=0.9)
    parser.add_argument('--S', type=int, default=5)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--IsDCN', type=int, default=0)
    parser.add_argument('--IsESM', type=int, default=0)
    parser.add_argument('--images_size', type=int, default=256)
    parser.add_argument('--images_path', type=str, default='/opt/data/Datasets/COCO/train')
    parser.add_argument('--save_models_path', type=str,
                        default='')
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--a', type=float, default=5)
    parser.add_argument('--b', type=float, default=5)
    parser.add_argument('--logdir', type=str,
                        default='')
    args = parser.parse_args()
    args.span_range_height = args.span_range_width = args.span_range
    args.grid_height = args.grid_width = args.grid_size
    args.image_height = args.image_width = args.images_size
    return args


def data(args):
    traindir = args.images_path
    if args.stego == 'DS':
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([args.images_size, args.images_size]),
                transforms.ToTensor(),
            ]))
    elif args.stego == 'ABDH':
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif args.stego == 'DAH':
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([256, 256], Image.BICUBIC),
                transforms.ToTensor(),
            ]))
    else:
        assert 1 == 1, 'the steganalysis is wrong!'

    assert train_dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    return train_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    # gpu
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    args = parseArgs()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    print('参数准备完成！')
    train_loader = data(args)
    print('train_loader准备完成！')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    if args.stego == 'DS':
        print('DS')
        Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
        Hnet.apply(weights_init)
        Rnet = RevealNet(output_function=nn.Sigmoid)
        Rnet.apply(weights_init)
        if args.Hnet != "":
            if (os.path.exists(args.Hnet)):
                Hnet.load_state_dict(torch.load(args.Hnet, map_location=device))
                print("成功从加载{}.pkl模型".format(args.Hnet))
            else:
                print('加载Hnet模型失败')
        if args.Rnet != '':
            if (os.path.exists(args.Rnet)):
                Rnet.load_state_dict(torch.load(args.Rnet, map_location=device))
                print("成功从加载{}.pkl模型".format(args.Rnet))
            else:
                print('加载Rnet模型失败')
        # 冻结Hne网络中的参数
        for name, parameter in Hnet.named_parameters():
            parameter.requires_grad = False
        # 冻结Rnet网络中的参数
        for name, parameter in Rnet.named_parameters():
            parameter.requires_grad = False
    elif args.stego == 'DAH':
        print("DAH")
        Hnet = UNet_FA_add_best(3, 3, bilinear=False, norm_layer=nn.InstanceNorm2d).to(device)
        Rnet = RevealNet_DAH(3, 3, nhf=64, norm_layer=nn.InstanceNorm2d, output_function=nn.Sigmoid).to(device)
        model_pth = torch.load(args.cp_path, map_location=device)
        Hnet.load_state_dict(model_pth['model_encoder'])
        Rnet.load_state_dict(model_pth['model_decoder'])
        Hnet.eval()
        Rnet.eval()
        Hnet.to(device)
        Rnet.to(device)
        # 冻结Hne网络中的参数
        for name, parameter in Hnet.named_parameters():
            parameter.requires_grad = False
        # 冻结Rnet网络中的参数
        for name, parameter in Rnet.named_parameters():
            parameter.requires_grad = False
    elif args.stego == 'ABDH':
        print("ABDH")
        attentionModel = ABDH.AttentionNet().to(device)
        Hnet = ABDH.GeneratorImage(7, 3).to(device)
        Rnet = ABDH.GeneratorImage(3, 3).to(device)
        model_pth = torch.load(args.cp_path, map_location=torch.device('cpu'))
        attentionModel.load_state_dict(model_pth['model_attention'])
        Hnet.load_state_dict(model_pth['model_encoder'])
        Rnet.load_state_dict(model_pth['model_decoder'])
        Hnet.eval()
        Rnet.eval()
        Hnet.to(device)
        Rnet.to(device)
        # 冻结Hne网络中的参数
        for name, parameter in Hnet.named_parameters():
            parameter.requires_grad = False
        # 冻结Rnet网络中的参数
        for name, parameter in Rnet.named_parameters():
            parameter.requires_grad = False
    else:
        assert 1 == 1, 'the steganalysis is wrong!'

    # # 冻结Hne网络中的参数
    # for name, parameter in Hnet.named_parameters():
    #     parameter.requires_grad = False
    # # 冻结Rnet网络中的参数
    # for name, parameter in Rnet.named_parameters():
    #     parameter.requires_grad = False

    # 形变网络
    model = STNTpsNet(args, device).to(device)
    # 优化器

    print('优化器准备完成！')
    # loss选择
    criterion = VGGLoss(device)
    print('loss准备完成！')
    # esm
    esm = estimateSigma(device).to(device)
    if args.stego == 'DS':
        myTrain = Train_ds(args=args, Hnet=Hnet, Rnet=Rnet, model=model, estimateSigma=esm, criterion=criterion,
                           device=device)
    elif args.stego == 'DAH':
        myTrain = Train_dah(args=args, Hnet=Hnet, Rnet=Rnet, model=model, estimateSigma=esm, criterion=criterion,
                            device=device)
    elif args.stego == 'ABDH':
        myTrain = Train_abdh(args=args, Hnet=Hnet, Rnet=Rnet, model=model, estimateSigma=esm, criterion=criterion,
                             device=device, attentionModel=attentionModel)
    else:
        assert 1 == 1, 'the steganalysis is wrong!'

    myTrain.train(train_loader)


if __name__ == '__main__':
    main()
