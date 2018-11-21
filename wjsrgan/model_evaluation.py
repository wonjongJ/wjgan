import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="celebA", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--hr_height', type=int, default=64, help='size of high res. image height')
parser.add_argument('--hr_width', type=int, default=64, help='size of high res. image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.hr_height / 2**4), int(opt.hr_width / 2**4)
patch = (opt.batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorResNet()

if cuda:
    generator = generator.cuda()

generator.load_state_dict(torch.load('/root/Jarvis/logs/wonjong/saved_models/2nd/generator_29.pth'))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_lr = Tensor(opt.batch_size, opt.channels, opt.hr_height//4, opt.hr_width//4)
input_hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
input_lr2hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)


# Transforms for low resolution images and high resolution images
lr_transforms = [   transforms.Resize((opt.hr_height//4, opt.hr_height//4), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

hr_transforms = [   transforms.Resize((opt.hr_height, opt.hr_height), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
lr2hr_transforms = [   transforms.Resize((opt.hr_height//4, opt.hr_height//4), Image.BICUBIC),
                    transforms.Resize((opt.hr_height, opt.hr_height), Image.BICUBIC), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]



dataloader = DataLoader(ImageDataset("/data1/wonjong/%s/test/" % opt.dataset_name, lr_transforms=lr_transforms, hr_transforms=hr_transforms, lr2hr_transforms = lr2hr_transforms),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

print("hello")

# ---------------------------
#  Using good initial guess
# ---------------------------
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(input_lr.copy_(imgs['lr']))


        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)


        batches_done = i
        # Save image sample
        imgs_lr2hr = Variable(input_lr2hr.copy_(imgs['lr2hr']))
        save_image(torch.cat((imgs_lr2hr.data, gen_hr.data), -2),
                    '/root/Jarvis/logs/wonjong/sample/result/celebA/%d.png' % batches_done, normalize=True)
