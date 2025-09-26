"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Created on 2025.09.26
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
import shutil
import argparse
import numpy as np

from dataset import TailingSensorSet
from util import AverageMeter
from sklearn.metrics import r2_score

from convlstm import ConvLSTM


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=30, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='CNNLSTM')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--resume', default=0, type=int, metavar='PATH', help='path to latest checkpoint')

    opt = parser.parse_args()

    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.temporal = ConvLSTM(input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=2, batch_first=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(518, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, var, ims):
        bsz, c, seq_len, h, w = ims.size()
        ims = ims.permute(0, 2, 1, 3, 4)
        ims = ims.reshape(bsz * seq_len, c, h, w)
        maps = self.feature(ims)
        _, c, h, w = maps.size()
        maps = maps.reshape(bsz, seq_len, c, h, w)
        codes = self.temporal(maps)
        codes = self.pool(codes)
        x = torch.cat((var.view(var.size(0), -1), codes.view(codes.size(0), -1)), dim=1)
        x = self.regressor(x)
        return x


def set_loader(opt):
    full_data = TailingSensorSet(train_mode="train", clip_mode='dynamic')
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size

    # train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_idx = list(range(0, train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, len(full_data)))
    train_data = torch.utils.data.Subset(full_data, train_idx)
    val_data = torch.utils.data.Subset(full_data, val_idx)
    test_data = torch.utils.data.Subset(full_data, test_idx)

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = CNNLSTM()
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
    for idx, (process_inputs, images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        process_inputs = process_inputs.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(process_inputs.float(), images)
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
        for idx, (process_inputs, images, targets) in enumerate(val_loader):
            process_inputs = process_inputs.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]

            # forward
            output = model(process_inputs.float(), images)
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
    tb = SummaryWriter(comment="CNNLSTM")
    opt.resume = "./save/CNNLSTM/checkpoint_{}.ckpt".format(opt.resume) if opt.resume else 0

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    if opt.resume:
        assert os.path.isfile(opt.resume), "=> no checkpoint found at '{}'".format(opt.resume)
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss_train = train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        loss_val = validate(val_loader, model, criterion, epoch, opt, tb)
        print('epoch {}, total time {:.2f}, loss_train {}, loss_val {}'.format(epoch, time2 - time1, loss_train, loss_val))

        if epoch % opt.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, "./save/DynaFormer/", epoch + 1)


def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint_{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)


if __name__ == '__main__':
    main()
