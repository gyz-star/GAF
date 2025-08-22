import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from GAF.model.GAF.grid_sample import grid_sample
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from GAF.model.GAF.DCT import DCT

def DT(image1, image2):
  I, J = image1.shape[2], image1.shape[3]

  diff = torch.abs(torch.sub(image1, image2))

  diff_sum = torch.sum(diff)
  result = diff_sum / (I * J*image1.shape[0]*3)
  return result

def save_model(save_path, iteration, optimizer, model):
    torch.save({'iteration': iteration,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
                save_path)
    print("model save success")

class Train:
    def __init__(self, args, Hnet, Rnet,estimateSigma, model, criterion, device,Threshold=0.4):
        self.estimateSigma = estimateSigma
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.mseloss = nn.MSELoss().to(device)
        self.Hnet = Hnet.to(device)
        self.Rnet = Rnet.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer_est=torch.optim.Adam(self.estimateSigma.parameters(),lr=args.lr)
        self.estimateSigma=estimateSigma
        self.height = args.images_size
        self.width = args.images_size
        self.Threshold=Threshold

    def train_batch(self,covers, secrets):
        self.optimizer.zero_grad()
        self.optimizer_est.zero_grad()
        self.model.train()
        # 首先读入数据利用预训练好的DS模型生成水印图像
        # 获得含密图像
        att = 0.0
        hidden = self.Hnet(secrets, covers, att)
        # 含密图像提取的秘密图像
        rev_img = self.Rnet(hidden)  # containerImg as input of R-net and get "rev_secret_img"


        #将hidden分为RGB
        hidden_R=hidden[:,0,:,:].unsqueeze(1)
        hidden_G=hidden[:,1,:,:].unsqueeze(1)
        hidden_B=hidden[:,2,:,:].unsqueeze(1)
        # 获得预测的通道选择
        c_R = self.estimateSigma(hidden_R)#batch 64
        # channels_R = torch.gt(channels_R, self.Threshold).float()
        c_G= self.estimateSigma(hidden_G)
        # channels_G = torch.gt(channels_G, self.Threshold).float()
        c_B = self.estimateSigma(hidden_B)
        # channels_B = torch.gt(channels_B, self.Threshold).float()
        channels_probability=torch.cat([c_R, c_G,c_B], 1)
        channels_probability=torch.unsqueeze(torch.unsqueeze(channels_probability, dim=2), dim=3)

        if self.args.IsESM == 0:
            channels_probability = torch.ones_like(channels_probability).to(self.device)
            channels_probability[:, 0:24, :, :] = 0
            channels_probability[:, 64:88, :, :] = 0
            channels_probability[:, 128:152, :, :] = 0
            print('不进行通道筛选！dah改！')

        #获得扭曲场
        grid= self.model(hidden)
        dct = DCT(device=self.device, N=8).to(self.device)
        image_dct = dct(hidden)  # B c*64 H/8 W/8
        #获取扭曲后的系数
        transformed_x_dct = grid_sample(image_dct, grid)
        transformed_x_dct1 = transformed_x_dct*channels_probability
        transformed_x_dct2=image_dct*(torch.ones_like(channels_probability).to(self.device)-channels_probability)
        transformed_x_dct=transformed_x_dct1+transformed_x_dct2
        #获取扭曲后的图像
        new_image = dct.reverse(transformed_x_dct)
        # 获取融合后的提取图像
        rev_img_new = self.Rnet(new_image)
        #####################loss#######################
        # Loss1： 使得扭曲后的高频与原始高频较为贴近
        # vggLoss=self.criterion(HomoImage, hidden_img)
        L1Loss = nn.L1Loss().to(self.device)
        # Loss1 = L1Loss(HomoImage, hidden_img)

        Loss1 = self.criterion(new_image, hidden)
        Loss1.requires_grad_(True)

        Loss2=1/L1Loss(rev_img,rev_img_new)
        # Loss3 = L1Loss(new_image, hidden_img)
        Loss = self.args.a*Loss1 + self.args.b*Loss2
        Loss.backward()
        self.optimizer_est.step()
        self.optimizer.step()

        # return Loss1.item(), Loss2.item(), Loss3,Loss.item(), x_dct, new_dct, new_image, hidden_img, rev_img, rev_img_new
        return  Loss1.item(),Loss2.item(),  new_image, hidden, rev_img, rev_img_new,c_R,c_G,c_B
    def train(self, train_dataloader):
        # 创建tensorboard
        writer = SummaryWriter(self.args.logdir)
        print('完成')
        SSIMImage = SSIM()
        SSIMImage.to(self.device)
        PSNRImage = PSNR()
        PSNRImage.to(self.device)
        num_data = 0
        tensorboard_time = 0
        for epochs in range(0, self.args.epochs):
            for step, data in enumerate(train_dataloader, 0):
                all_pics = data  # allpics contains cover images and secret images
                this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

                # first half of images will become cover images, the rest are treated as secret images
                cover_img = all_pics[0:this_batch_size, :, :, :].to(self.device)  # batchSize,3,256,256
                secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :].to(self.device)
                # Loss1,Loss2, Loss3,  Loss, x_dct, new_dct, new_image, hidden_img, rev_img, rev_img_new= self.train_batch(
                #     cover_img, secret_img)
                Loss1,Loss2, new_image, hidden_img, rev_img, rev_img_new,c_R,c_G,c_B = self.train_batch(
                    cover_img, secret_img)
                with torch.no_grad():
                    num_data += 1
                    if num_data % self.args.log_step == 0:
                        tensorboard_time += 1
                        # writer.add_scalar('Loss', Loss, tensorboard_time)
                        writer.add_scalar('Loss1', Loss1, tensorboard_time)
                        writer.add_scalar('Loss2', Loss2, tensorboard_time)
                        # writer.add_scalar('Loss3', Loss3, tensorboard_time)
                        writer.add_scalar('image_ssim', SSIMImage(hidden_img, new_image), tensorboard_time)
                        writer.add_scalar('image_psnr', PSNRImage(hidden_img, new_image), tensorboard_time)
                        writer.add_scalar('DT', DT(rev_img.cpu(), rev_img_new.cpu()), tensorboard_time)
                        # # 整体图像
                        # writer.add_images('hidden', hidden_img, tensorboard_time)
                        # writer.add_images('new_images', new_image, tensorboard_time)
                        # writer.add_images('difference', hidden_img - new_image, tensorboard_time)
                        # # 提取图像
                        # writer.add_images('rev_img', rev_img, tensorboard_time)
                        # writer.add_images('rev_img_new', rev_img_new, tensorboard_time)
                        if tensorboard_time == 1:
                            with open(self.args.logdir+'channels.txt', 'w') as file:
                                file.write('=================================tensorboard_time:{}====================================='.format(tensorboard_time))
                                file.write('\r\n')
                                file.write('c_R: {}'.format(c_R))
                                file.write('\r\n')
                                file.write('c_G: {}'.format(c_G))
                                file.write('\r\n')
                                file.write('c_B: {}'.format(c_B))
                                file.write('\r\n')

                        else:
                            with open(self.args.logdir+'channels.txt', 'a') as file:
                                file.write('=================================tensorboard_time:{}====================================='.format(tensorboard_time))
                                file.write('\r\n')
                                file.write('c_R: {}'.format(c_R))
                                file.write('\r\n')
                                file.write('c_G: {}'.format(c_G))
                                file.write('\r\n')
                                file.write('c_B: {}'.format(c_B))
                                file.write('\r\n')
                        print('记录一次!')
            # 保存stn模型参数
            save_model(self.args.save_models_path + 'model{}.pth'.format(epochs), epochs, self.optimizer, self.model)
            save_model(self.args.save_models_path + 'estimateSigma_model{}.pth'.format(epochs), epochs,
                       self.optimizer_est, self.estimateSigma)

class Train_32:
    def __init__(self, args, Hnet, Rnet,estimateSigma, model, criterion, device,Threshold=0.4):
        self.estimateSigma = estimateSigma
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.mseloss = nn.MSELoss().to(device)
        self.Hnet = Hnet.to(device)
        self.Rnet = Rnet.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer_est=torch.optim.Adam(self.estimateSigma.parameters(),lr=args.lr)
        self.estimateSigma=estimateSigma
        self.height = args.images_size
        self.width = args.images_size
        self.Threshold=Threshold

    def train_batch(self,covers, secrets):
        self.optimizer.zero_grad()
        self.optimizer_est.zero_grad()
        self.model.train()
        # 首先读入数据利用预训练好的DS模型生成水印图像
        # 获得含密图像
        att = 0.0
        hidden = self.Hnet(secrets, covers, att)
        # 含密图像提取的秘密图像
        rev_img = self.Rnet(hidden)  # containerImg as input of R-net and get "rev_secret_img"


        #将hidden分为RGB
        hidden_R=hidden[:,0,:,:].unsqueeze(1)
        hidden_G=hidden[:,1,:,:].unsqueeze(1)
        hidden_B=hidden[:,2,:,:].unsqueeze(1)
        # 获得预测的通道选择
        c_R = self.estimateSigma(hidden_R)#batch 64
        # channels_R = torch.gt(channels_R, self.Threshold).float()
        c_G= self.estimateSigma(hidden_G)
        # channels_G = torch.gt(channels_G, self.Threshold).float()
        c_B = self.estimateSigma(hidden_B)
        # channels_B = torch.gt(channels_B, self.Threshold).float()
        channels_probability=torch.cat([c_R, c_G,c_B], 1)
        channels_probability=torch.unsqueeze(torch.unsqueeze(channels_probability, dim=2), dim=3)


        #获得扭曲场
        grid= self.model(hidden)
        dct = DCT(device=self.device, N=8).to(self.device)
        image_dct = dct(hidden)  # B c*64 H/8 W/8
        #获取扭曲后的系数
        transformed_x_dct = grid_sample(image_dct, grid)
        transformed_x_dct1 = transformed_x_dct*channels_probability
        transformed_x_dct2=image_dct*(torch.ones_like(channels_probability).to(self.device)-channels_probability)
        transformed_x_dct=transformed_x_dct1+transformed_x_dct2
        #获取扭曲后的图像
        new_image = dct.reverse(transformed_x_dct)
        # 获取融合后的提取图像
        rev_img_new = self.Rnet(new_image)
        #####################loss#######################
        # Loss1： 使得扭曲后的高频与原始高频较为贴近
        # vggLoss=self.criterion(HomoImage, hidden_img)
        L1Loss = nn.L1Loss().to(self.device)
        # Loss1 = L1Loss(HomoImage, hidden_img)

        Loss1 = self.criterion(new_image, hidden)
        Loss1.requires_grad_(True)

        Loss2=1/L1Loss(rev_img,rev_img_new)
        # Loss3 = L1Loss(new_image, hidden_img)
        Loss = self.args.a*Loss1 + self.args.b*Loss2
        Loss.backward()
        self.optimizer_est.step()
        self.optimizer.step()

        # return Loss1.item(), Loss2.item(), Loss3,Loss.item(), x_dct, new_dct, new_image, hidden_img, rev_img, rev_img_new
        return  Loss1.item(),Loss2.item(),  new_image, hidden, rev_img, rev_img_new,c_R,c_G,c_B
    def train(self, train_dataloader):
        # 创建tensorboard
        writer = SummaryWriter(self.args.logdir)
        print('完成')
        SSIMImage = SSIM()
        SSIMImage.to(self.device)
        PSNRImage = PSNR()
        PSNRImage.to(self.device)
        num_data = 0
        tensorboard_time = 0
        for epochs in range(0, self.args.epochs):
            for step, (data,_) in enumerate(train_dataloader, 0):
                all_pics = data  # allpics contains cover images and secret images
                this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

                # first half of images will become cover images, the rest are treated as secret images
                cover_img = all_pics[0:this_batch_size, :, :, :].to(self.device)  # batchSize,3,256,256
                secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :].to(self.device)
                # Loss1,Loss2, Loss3,  Loss, x_dct, new_dct, new_image, hidden_img, rev_img, rev_img_new= self.train_batch(
                #     cover_img, secret_img)
                Loss1,Loss2, new_image, hidden_img, rev_img, rev_img_new,c_R,c_G,c_B = self.train_batch(
                    cover_img, secret_img)
                with torch.no_grad():
                    num_data += 1
                    if num_data % self.args.log_step == 0:
                        tensorboard_time += 1
                        # writer.add_scalar('Loss', Loss, tensorboard_time)
                        writer.add_scalar('Loss1', Loss1, tensorboard_time)
                        writer.add_scalar('Loss2', Loss2, tensorboard_time)
                        # writer.add_scalar('Loss3', Loss3, tensorboard_time)
                        writer.add_scalar('image_ssim', SSIMImage(hidden_img, new_image), tensorboard_time)
                        writer.add_scalar('image_psnr', PSNRImage(hidden_img, new_image), tensorboard_time)
                        writer.add_scalar('DT', DT(rev_img.cpu(), rev_img_new.cpu()), tensorboard_time)
                        # 整体图像
                        writer.add_images('hidden', hidden_img, tensorboard_time)
                        writer.add_images('new_images', new_image, tensorboard_time)
                        writer.add_images('difference', hidden_img - new_image, tensorboard_time)
                        # 提取图像
                        writer.add_images('rev_img', rev_img, tensorboard_time)
                        writer.add_images('rev_img_new', rev_img_new, tensorboard_time)
                        if tensorboard_time == 1:
                            with open(self.args.logdir+'channels.txt', 'w') as file:
                                file.write('=================================tensorboard_time:{}====================================='.format(tensorboard_time))
                                file.write('\r\n')
                                file.write('c_R: {}'.format(c_R))
                                file.write('\r\n')
                                file.write('c_G: {}'.format(c_G))
                                file.write('\r\n')
                                file.write('c_B: {}'.format(c_B))
                                file.write('\r\n')

                        else:
                            with open(self.args.logdir+'channels.txt', 'a') as file:
                                file.write('=================================tensorboard_time:{}====================================='.format(tensorboard_time))
                                file.write('\r\n')
                                file.write('c_R: {}'.format(c_R))
                                file.write('\r\n')
                                file.write('c_G: {}'.format(c_G))
                                file.write('\r\n')
                                file.write('c_B: {}'.format(c_B))
                                file.write('\r\n')
                        print('记录一次!')
            # 保存stn模型参数
            torch.save(self.model.state_dict(), self.args.save_models_path + 'model{}.pth'.format(epochs))
            torch.save(self.estimateSigma.state_dict(), self.args.save_models_path + 'estimateSigma_model{}.pth'.format(epochs))