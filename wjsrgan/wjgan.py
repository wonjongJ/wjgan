"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

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
parser.add_argument('--hr_dataset_name', type=str, default="CelebA", help='name of the dataset')
parser.add_argument('--lr_dataset_name', type=str, default="wider_lnew", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.0, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
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
patch = (opt.batch_size, 1, 1, 1)

# Initialize generator and discriminator
generator_h2l = GeneratorResNet_2()
discriminator_h2l = Discriminator()
feature_extractor = FeatureExtractor()

# Losses
criterion_GAN = torch.nn.HingeEmbeddingLoss(size_average=True)
criterion_content = torch.nn.MSELoss()

if cuda:
    generator_h2l = generator_h2l.cuda()
    discriminator_h2l = discriminator_h2l.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator_h2l.load_state_dict(torch.load('/root/Jarvis/logs/wonjong/saved_models/2nd/generator_h2l_0.pth'))
else:
    # Initialize weights
    generator_h2l.apply(weights_init_normal)
    discriminator_h2l.apply(weights_init_normal)

# Optimizers
optimizer_G_h2l = torch.optim.Adam(generator_h2l.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_h2l = torch.optim.Adam(discriminator_h2l.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_lr = Tensor(opt.batch_size, opt.channels, opt.hr_height//4, opt.hr_width//4)
input_hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
input_lr2hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
input_hr2lr = Tensor(opt.batch_size, opt.channels, opt.hr_height//4, opt.hr_width//4)


# Adversarial ground truths
fake = Variable(Tensor(np.ones(patch)), requires_grad=False)
valid = Variable(Tensor(-1*np.ones(patch)), requires_grad=False)

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



dataloader = DataLoader(ImageDataset("/data1/wonjong/Dataset/HIGH/%s/" % opt.hr_dataset_name, "/data1/wonjong/Dataset/LOW/%s/" % opt.lr_dataset_name, lr_transforms=lr_transforms, hr_transforms=hr_transforms, lr2hr_transforms = lr2hr_transforms),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

print("hello")

# ---------------------------
#  Using good initial guess
# ---------------------------
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = Variable(input_lr.copy_(imgs['lr']))
        imgs_hr = Variable(input_hr.copy_(imgs['hr']))
        imgs_hr2lr = Variable(input_hr2lr.copy_(imgs['hr2lr']))

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G_h2l.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_lr = generator_h2l(imgs_hr)

        # Adversarial loss
        gen_validity = discriminator_h2l(gen_lr)
        loss_GAN = criterion_GAN(gen_validity, valid)

        # # Content loss
        # gen_features = feature_extractor(gen_hr)
        # real_features = Variable(feature_extractor(imgs_hr).data, requires_grad=False)
        # loss_content =  criterion_content(gen_features, real_features)

        # Content loss
        loss_content = criterion_content(gen_lr, imgs_hr2lr)

        # Total loss
        loss_G = loss_content - 0.05 * loss_GAN

        loss_G.backward()
        optimizer_G_h2l.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_h2l.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator_h2l(imgs_lr), valid) # imgs_hr -> dataset hr img
        loss_fake = criterion_GAN(discriminator_h2l(gen_lr.detach()), fake)
        
        print(discriminator_h2l(imgs_lr))
        print(discriminator_h2l(gen_lr.detach()))
        # Total loss
        loss_D = - loss_real -(1+ loss_fake)
        print(loss_real)
        print(loss_fake)

        loss_D.backward()
        optimizer_D_h2l.step()

        # --------------
        #  Log Progress
        # --------------

        # print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
        #                                                     (epoch, opt.n_epochs, i, len(dataloader),
        #                                                      loss_G.item()))
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                            (epoch, opt.n_epochs, i, len(dataloader),
                                                            loss_D.item(), loss_G.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image sample
            save_image(torch.cat((gen_lr.data, imgs_hr2lr.data), -2),
                        '/root/Jarvis/logs/wonjong/sample/%d.png' % batches_done, normalize=True)



    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    # Save model checkpoints
        torch.save(generator_h2l.state_dict(), '/root/Jarvis/logs/wonjong/saved_models/2nd/generator_h2l_%d.pth' % epoch)
        torch.save(discriminator_h2l.state_dict(), '/root/Jarvis/logs/wonjong/saved_models/2nd/discriminator_h2l_%d.pth' % epoch)
