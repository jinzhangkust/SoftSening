import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.backends.cudnn as cudnn

import os
import time
import numpy as np
import pandas as pd
from PIL import Image

from embedding_networks_end2end import TransformerEmbedding
from util import AverageMeter, cal_accuracy


class test(Dataset):
    def __init__(self, model): # csv_file = 'XRFImgData4FeedRegression.csv'
        root = '/media/neuralits/Data_SSD/FrothData/Data4FrothGrade'
        csv_file = '/media/neuralits/Data_SSD/FrothData/XRFImgData4FeedRegression.csv'

        df=pd.read_csv(csv_file)
        tailing = df.iloc[:,0:3].values
        feed = df.iloc[:,4:7].values
        clip_db = df.iloc[:,3].values

        mean_tailing = [tailing[:,0].mean(), tailing[:,1].mean(), tailing[:,2].mean()]
        std_tailing = [tailing[:,0].std(), tailing[:,1].std(), tailing[:,2].std()]
        self.tailing = (tailing - mean_tailing) / std_tailing
        mean_feed = [feed[:,0].mean(), feed[:,1].mean(), feed[:,2].mean()]
        std_feed = [feed[:,0].std(), feed[:,1].std(), feed[:,2].std()]
        self.feed = (feed - mean_feed) / std_feed

        normalize = torchvision.transforms.Normalize(mean=[0.5429, 0.5580, 0.5357],
                                                          std=[0.1841, 0.1923, 0.2079])
        transform_clip = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize])

        for idx in range(len(df)):
            tailing = torch.tensor( self.tailing[idx,:], dtype=torch.float64 )
            truth = torch.tensor( self.feed[idx,:], dtype=torch.float64 )
            clip = clip_db[idx]
            
            time_stamp = clip[:14]
            img_list = torch.FloatTensor(3, 10, 400, 400) # [channels, frames, height, width]
            for i in range(1, 11):
                file_name = "{}_{}.jpg".format(time_stamp, i)
                full_img_path = os.path.join(root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i-1, :, :] = transform_clip(img).float()
            top = np.random.randint(0, 100)
            left = np.random.randint(0, 100)
            images = img_list[:, :, top : top + 300, left : left + 300]
            
            images = images.cuda()

            tailing_model = tailing.unsqueeze(0).cuda()
            pred = model(tailing_model.float(), images.unsqueeze(0))

            if idx:
                #features = torch.cat((features, feature.detach()), dim=0)
                predict_set = np.append(predict_set, pred.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, truth.numpy(), axis=0)
            else:
                #features = feature.detach()
                predict_set = pred.detach().cpu().numpy()
                target_set = truth.numpy()
        #self.features = features.cpu()
        acc0, acc1, acc2 = cal_accuracy(predict_set, target_set.reshape(predict_set.shape[0],3))
        print(f"Acc0: {acc0}    Acc1: {acc1}    Acc2: {acc2}")
        
    
model = TransformerEmbedding(num_layers=3, d_model=2048, num_heads=8, conv_hidden_dim=1024, emb_dim=3)
save_folder = '/media/neuralits/Data_SSD/Programs/PyTorch/FeedGradeEstimation/save/TransformerEmbed'
save_file = os.path.join(save_folder, 'transformer_embed_epoch_150.pth')
model.load_state_dict(torch.load(save_file))
model = model.cuda()
model.eval()
test(model)