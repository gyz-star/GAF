import argparse
import os
import sys
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from tensorboardX import SummaryWriter
sys.path.append(r"/opt/data/helin/Code/")
import new_stn.transformed as transforms
from PIL import Image
import torchvision

transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])
CI = r'/opt/data/helin/Code/new_stn/test_image/test_DS/images_32/image1.jpg'
SI = r'/opt/data/helin/Code/new_stn/test_image/test_DS/images_32/image2.jpg'
output_path = '/opt/data/helin/Code/pixelSteganalysis_3/myResult/'
cover_img = transform(Image.open(CI).convert('RGB')).unsqueeze(0)
secret_img = transform(Image.open(SI).convert('RGB')).unsqueeze(0)
torchvision.utils.save_image(cover_img[:,:,0:32,0:32], output_path  + 'cover_img_0.jpg')
torchvision.utils.save_image(secret_img[:,:,0:32,0:32], output_path  + 'secret_img_0.jpg')
