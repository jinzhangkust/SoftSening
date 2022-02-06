__author__ = 'Jin Zhang'

import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
import random
import os

#定义一个数据集
class XRFImg4FeedSet(Dataset):
    def __init__(self, root, csv_file, train_mode, clip_mode): # csv_file = 'XRFImgData4FeedRegression.csv'
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.clip_mode = clip_mode
        self.root = root
        self.df=pd.read_csv(csv_file)
        tailing = self.df.iloc[:,0:3].values
        feed = self.df.iloc[:,4:7].values
        clip = self.df.iloc[:,3].values
        
        mean_tailing = [tailing[:,0].mean(), tailing[:,1].mean(), tailing[:,2].mean()]
        std_tailing = [tailing[:,0].std(), tailing[:,1].std(), tailing[:,2].std()]
        tailing = (tailing - mean_tailing) / std_tailing
        mean_feed = [feed[:,0].mean(), feed[:,1].mean(), feed[:,2].mean()]
        std_feed = [feed[:,0].std(), feed[:,1].std(), feed[:,2].std()]
        #print(f'mean_feed: {mean_feed}    std_feed: {std_feed}')
        feed = (feed - mean_feed) / std_feed
        
        index = np.random.RandomState(seed=56).permutation(len(self.df)) #np.random.permutation(len(self.df))
        self.tailing = tailing[index,:]
        self.feed = feed[index,:]
        self.clip = clip[index]
        #print(f'self.clip: {self.clip}')
        transform = None
        if transform is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491],
                                                          std=[0.1833, 0.1916, 0.2061])
            if train_mode == "train":
                self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(300),
                torchvision.transforms.ToTensor(),
                normalize])
            else:
                self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(300),
                torchvision.transforms.ToTensor(),
                normalize])
                
        transform_clip = None
        if transform_clip is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5429, 0.5580, 0.5357],
                                                          std=[0.1841, 0.1923, 0.2079])
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
        tailing = torch.tensor( self.tailing[idx,:], dtype=torch.float64 )
        truth = torch.tensor( self.feed[idx,:], dtype=torch.float64 )
        clip = self.clip[idx]
        #print(f'clip: {clip}')
        
        time_stamp = clip[:14]
        #print('time_stamp_1: {}'.format(time_stamp))
        if self.clip_mode == "single":
            file_name = "{}_{}.jpg".format(time_stamp, random.randint(1,10))
            full_img_path = os.path.join(self.root, clip, file_name)
            """for i in range(len(clip)):
                time_stamp = clip[i] #__getitem__一次性只读取一张图像，因此这种索引实际不是batch里的第i个元素，而是time_stamp的第i位
                print('time_stamp_1: {}'.format(time_stamp))
                time_stamp = time_stamp[:14]
                print(f'time_stamp_2: {time_stamp}')
                file_name = "{}_{}.jpg".format(time_stamp, random.randint(1,10))
                full_img_path = os.path.join(self.root, clip[i], file_name)"""
            images = Image.open(full_img_path).convert("RGB")
            images = self.transforms(images)
            
        else:
            #print("==> Hello, I am here.")
            img_list = torch.FloatTensor(3, 10, 400, 400) # [channels, frames, height, width]
            for i in range(1, 11):
                file_name = "{}_{}.jpg".format(time_stamp, i)
                full_img_path = os.path.join(self.root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i-1, :, :] = self.transform_clip(img).float()
            top = np.random.randint(0, 100)
            left = np.random.randint(0, 100)
            images = img_list[:, :, top : top + 300, left : left + 300]
        
        return tailing, images, truth
    
    