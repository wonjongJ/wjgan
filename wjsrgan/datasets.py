import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_hr1, root_hr2, root_hr3, root_lr, lr_transforms=None, hr_transforms=None, lr2hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.lr2hr_transform = transforms.Compose(lr2hr_transforms)

        hr_list1 = glob.glob(root_hr1 + '/*.*')
        hr_list2 = glob.glob(root_hr2 + '/*.*')
        hr_list3 = glob.glob(root_hr3 + '/*.*')

        self.hr_files = sorted(hr_list1+hr_list2+hr_list3)
        self.lr_files = sorted(glob.glob(root_lr + '/*.*'))


    def __getitem__(self, index):
        hr_img = Image.open(self.hr_files[index % len(self.hr_files)])
        lr_img = Image.open(self.lr_files[index % len(self.lr_files)])
        img_lr = self.lr_transform(lr_img)
        img_lr2hr = self.lr_transform(lr_img) # to be determined.
        img_hr2lr = self.lr_transform(hr_img)
        # img_lr2hr = self.lr2hr_transform(img)
        img_hr = self.hr_transform(hr_img)

        return {'lr': img_lr, 'hr': img_hr, 'lr2hr': img_lr2hr, 'hr2lr' : img_hr2lr}

    def __len__(self):
        return len(self.hr_files) # to be considered.
