# coding=UTF-8
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, mean_squared_error, \
    mean_absolute_error
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr

from env import *


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')  # 可以改为sum试试结果

    return loss


def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    xBar = targets.mean()
    yBar = outputs.mean()
    SSR = 0
    varX = 0  # 公式中分子部分
    varY = 0  # 公式中分母部分
    for i in range(0, targets.shape[0]):
        diffXXBar = targets[i] - xBar
        diffYYBar = outputs[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = torch.sqrt(varX * varY)
    xxx = SSR / SST
    return torch.mean(xxx)


def rmse(preds, labels):
    loss = (preds - labels) ** 2
    loss = torch.mean(loss)
    return torch.sqrt(loss)


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    return torch.mean(loss)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model=None, save_path='', args=None, train_dataloader=None, val_dataloader=None, train_scale_y=None,
          val_scale_y=None):
    seed = args.random_seed

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    min_loss = 1e+8

    i = 0
    epoch = args.epoch
    print(epoch)
    early_stop_win = 15

    model.train()

    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        t_train_predicted_list = []
        t_train_ground_list = []
        model.train()

        j = 0
        for speed, image, labels in dataloader:
            model.train()
            _start = time.time()
            speed, image, labels = [item.float().to(device) for item in [speed, image, labels]]

            out = model(image, speed).float().to(device)

            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())

            if len(t_train_predicted_list) <= 0:
                t_train_predicted_list = out
                t_train_ground_list = labels
            else:
                t_train_predicted_list = torch.cat((t_train_predicted_list, out), dim=0)
                t_train_ground_list = torch.cat((t_train_ground_list, labels), dim=0)

            i += 1
            j += 1

        t_train_predicted_list = train_scale_y.inverse_transform(t_train_predicted_list)
        t_train_ground_list = train_scale_y.inverse_transform(t_train_ground_list)

        rmse_all = rmse(t_train_predicted_list, t_train_ground_list)
        mae_all = mae(t_train_predicted_list, t_train_ground_list)
        train_cc = computecc(t_train_predicted_list, t_train_ground_list)

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, RMSE:{:.8f}, MAE:{:.8f}, CC:{:.8f}, lr:{:.8f})'.format(
            i_epoch, epoch,
            rmse_all / len(dataloader), rmse_all, mae_all, train_cc, get_lr(optimizer)),
            flush=True
        )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result, val_rmse, val_mae, val_cc = val(model, val_dataloader, val_scale_y)

            print('val : (Loss:{:.8f},  RMSE:{:.8f}, MAE:{:.8f}, CC:{:.8f})'.format(val_loss, val_rmse, val_mae, val_cc),
                  flush=True)

            if val_loss < min_loss:
                # torch.save(model.state_dict(), save_path)
                torch.save(model, save_path)
                print("save best model at ", save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if rmse_all < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = rmse_all

    return train_loss_list
