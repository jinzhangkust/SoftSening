"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Modified from "Using a Novel Clustered 3D-CNN Model for Improving Crop Future Price Prediction" in KBS 2023
Created on 2023.08.04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from dataset import TailingSensorSet
from util import AverageMeter
from sklearn.metrics import r2_score

from wideresnet3D import WideResNet3D


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='3DNet-2-3')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')

    opt = parser.parse_args()

    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


class model3D(nn.Module):
    def __init__(self):
        super(model3D, self).__init__()
        self.feature = WideResNet3D()
        #self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(515, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x0, x1):
        #print(f"shape of x1: {x1.shape}")
        x1 = self.feature(x1)
        #print(f"shape of x1: {x1.shape}")
        #x1 = self.pool(x1)
        #print(f"shape of x1: {x1.shape}")
        x = torch.cat((x0.view(x0.size(0), -1), x1.view(x1.size(0), -1)), dim=1)
        x = self.regressor(x)
        return x


def set_loader(opt):
    full_data = TailingSensorSet(train_mode="train", clip_mode='seq')
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = model3D()

    criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer


def cal_accuracy(out_hat, truth_hat):
    """mean = [1.3957, 0.4854, 25.4088]
    std = [0.2851, 0.0348, 1.0076]
    out = out_hat*std + mean
    truth = truth_hat*std + mean
    R2_0 = r2_score(out[:,0], truth[:,0])
    R2_1 = r2_score(out[:,1], truth[:,1])
    R2_2 = r2_score(out[:,2], truth[:,2])"""
    R2_0 = r2_score(out_hat[:, 0], truth_hat[:, 0])
    R2_1 = r2_score(out_hat[:, 1], truth_hat[:, 1])
    R2_2 = r2_score(out_hat[:, 2], truth_hat[:, 2])
    return R2_0, R2_1, R2_2


def warmup_learning_rate(opt, epoch, idx, nBatch, optimizer):
    T_total = opt.epochs * nBatch
    T_warmup = 10 * nBatch
    if epoch <= 10 and idx <= T_warmup:
        lr = 1e-6 + (opt.learning_rate - 1e-6) * idx / T_warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, opt, tb):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_loss = 0

    end = time.time()
    for idx, (reagents, images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        reagents = reagents.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(reagents.float(), images)
        loss = criterion(output, targets.float())

        # update metric
        if idx == 0:
            predict_set = output.detach().cpu().numpy()
            target_set = targets.cpu().numpy()
        else:
            predict_set = np.append(predict_set, output.detach().cpu().numpy(), axis=0)
            target_set = np.append(target_set, targets.cpu().numpy(), axis=0)

        losses.update(loss.item(), bsz)
        total_loss += loss.item()
        acc0, acc1, acc2 = cal_accuracy(predict_set, target_set)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'R-square {acc:.3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc0))
            sys.stdout.flush()

    # grid = torchvision.utils.make_grid(images)
    # tb.add_image("images", grid)
    # tb.add_graph(model, (tailings.float(),images))
    # acc0, acc1, acc2 = cal_accuracy(predict_set, target_set)
    tb.add_scalar("Acc0", acc0, epoch)
    tb.add_scalar("Acc1", acc1, epoch)
    tb.add_scalar("Acc2", acc2, epoch)
    tb.add_scalar("Loss", total_loss, epoch)

    return losses.avg


def validate(val_loader, model, criterion, epoch, opt, tb):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    total_loss = 0

    with torch.no_grad():
        end = time.time()
        for idx, (reagents, images, targets) in enumerate(val_loader):
            reagents = reagents.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]

            # forward
            output = model(reagents.float(), images)
            loss = criterion(output, targets.float())

            if idx:
                predict_set = np.append(predict_set, output.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            else:
                predict_set = output.detach().cpu().numpy()
                target_set = targets.cpu().numpy()

            # update metric
            losses.update(loss.item(), bsz)
            total_loss += loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses))

    acc0, acc1, acc2 = cal_accuracy(predict_set, target_set)
    tb.add_scalar("Test-Acc0", acc0, epoch)
    tb.add_scalar("Test-Acc1", acc1, epoch)
    tb.add_scalar("Test-Acc2", acc2, epoch)
    tb.add_scalar("Test-Loss", total_loss, epoch)

    return losses.avg


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="3DNet-2-3")

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss_train = train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        loss_val = validate(val_loader, model, criterion, epoch, opt, tb)
        print('epoch {}, total time {:.2f}, loss_train {}, loss_val {}'.format(epoch, time2 - time1, loss_train,
                                                                               loss_val))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, '3DNet_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
