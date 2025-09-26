"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2023.08.02
"""

import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image, ImageFile, ImageFilter
import pandas as pd
import random
import os


normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491], std=[0.1833, 0.1916, 0.2061])


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandAugment:
    def __init__(self, k):
        self.k = k
        self.augment_pool = [torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=0.7),
            torchvision.transforms.RandomHorizontalFlip()]
    def __call__(self, im):
        ops = random.choices(self.augment_pool, k=self.k)
        for op in ops:
            if random.random() < 0.5:
                im = op(im)
        return im


class TransformTwice:
    def __init__(self, imsize):
        self.transform_weak = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(imsize, scale=((imsize-10)/400, (imsize+10)/400)),
            torchvision.transforms.ToTensor(),
            normalize])
        self.transform_strong = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(imsize, scale=((imsize-10)/400, (imsize+10)/400)),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=0.7),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize])
    def __call__(self, x):
        return [self.transform_weak(x), self.transform_strong(x)]


class TailingSensorSet(Dataset):
    def __init__(self, train_mode, clip_mode, frames=8, imsize=300):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.root = '/home/libsv/Data/FrothClips'
        csv_file = '/home/libsv/Data/FeedImgReagent4Tail_Sort.csv'
        self.clip_mode = clip_mode
        self.frames = frames
        self.imsize = imsize

        self.df = pd.read_csv(csv_file)
        feed = self.df.iloc[:, 0:3].values
        self.clip = self.df.iloc[:, 3].values
        reagent = self.df.iloc[:, 4:7].values
        tailing = self.df.iloc[:, 7:10].values
        tailing = tailing / tailing.max(0)

        process_inputs = [feed, reagent]

        
        mean_feed = [feed[:, 0].mean(), feed[:, 1].mean(), feed[:,2 ].mean()]
        std_feed = [feed[:, 0].std(), feed[:, 1].std(), feed[:, 2].std()]
        feed = (feed - mean_feed) / std_feed
        print(f"mean_feed: {mean_feed} \n std_feed: {std_feed}")
        mean_reagent = [reagent[:, 0].mean(), reagent[:, 1].mean(), reagent[:, 2].mean()]
        std_reagent = [reagent[:, 0].std(), reagent[:, 1].std(), reagent[:, 2].std()]
        reagent = (reagent - mean_reagent) / std_reagent
        print(f"mean_reagent: {mean_reagent} \n std_reagent: {std_reagent}")
        self.process_inputs = np.concatenate((feed, reagent), axis=1)

        mean_tailing = [tailing[:, 0].mean(), tailing[:, 1].mean(), tailing[:, 2].mean()]
        std_tailing = [tailing[:, 0].std(), tailing[:, 1].std(), tailing[:, 2].std()]
        print(f"mean_tailing: {mean_tailing} \n std_tailing: {std_tailing}")
        self.tailing = (tailing - mean_tailing) / std_tailing

        transform = None
        if transform is None:
            if train_mode == "train":
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(self.imsize, scale=((self.imsize - 10) / 400, (self.imsize + 10) / 400)),
                    RandAugment(k=3),
                    torchvision.transforms.ToTensor(),
                    normalize])
            else:
                self.transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(300),
                torchvision.transforms.ToTensor(),
                normalize])
                
        transform_clip = None
        if transform_clip is None:
            self.transform_clip = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize])
        
    def denormalize4img(self, x_hat):
        mean = [0.5561, 0.5706, 0.5491]
        std = [0.1833, 0.1916, 0.2061]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __len__(self):
        return len(self.clip)

    def __getitem__(self, idx):
        truth = torch.tensor(self.tailing[idx, :], dtype=torch.float32)
        clip = self.clip[idx]
        process_input = torch.tensor(self.process_inputs[idx, :], dtype=torch.float32)

        if self.clip_mode == "dynamic":
            img_list = torch.FloatTensor(3, self.frames, 400, 400)  # [channels, frames, height, width]
            for i in range(1, self.frames + 1):
                file_name = "libsv{}.jpg".format(i)
                full_img_path = os.path.join(self.root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i - 1, :, :] = self.transform_clip(img).float()
            top = np.random.randint(0, 100)
            left = np.random.randint(0, 100)
            images = img_list[:, :, top: top + 300, left: left + 300]
        elif self.clip_mode == "full":
            img_list = torch.FloatTensor(3, self.frames, 400, 400)  # [channels, frames, height, width]
            for i in range(1, self.frames + 1):
                file_name = "libsv{}.jpg".format(i)
                full_img_path = os.path.join(self.root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i - 1, :, :] = self.transform_clip(img).float()
            top = np.random.randint(0, 100)
            left = np.random.randint(0, 100)
            seq = img_list[:, :, 50: 50 + 300, 50: 50 + 300]

            file_name = "libsv{}.jpg".format(random.randint(1, 15))
            full_img_path = os.path.join(self.root, clip, file_name)
            single = Image.open(full_img_path).convert("RGB")
            single = self.transform(single)
            images = (single, seq)
        else:
            file_name = "libsv{}.jpg".format(random.randint(1, 15))
            full_img_path = os.path.join(self.root, clip, file_name)
            images = Image.open(full_img_path).convert("RGB")
            images = self.transform(images)
        
        return process_input, images, truth
