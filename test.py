import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from env import *


def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    xBar = targets.mean()
    yBar = outputs.mean()
    # print(xBar,yBar)
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


def val(model, dataloader, test_scale):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    t_test_predicted_list = []
    t_test_ground_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0

    for speed, image, y in dataloader:
        speed, image, y = [item.to(device).float() for item in [speed, image, y]]

        with torch.no_grad():
            predicted = model(image, speed).float().to(device)

            loss = loss_func(predicted, y)

            # print("test_x, predicted, y", x, predicted, y)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)

        test_loss_list.append(loss.item())

        i += 1
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    t_test_predicted_list = test_scale.inverse_transform(t_test_predicted_list)
    t_test_ground_list = test_scale.inverse_transform(t_test_ground_list)
    val_cc = computecc(t_test_predicted_list, t_test_ground_list)
    val_rmse = rmse(t_test_predicted_list, t_test_ground_list)
    val_mae = mae(t_test_predicted_list, t_test_ground_list)
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list], val_rmse, val_mae, val_cc


def test(model, dataloader, test_scale):
    # test data
    device = get_device()
    t_test_predicted_list = []
    t_test_ground_list = []

    for speed, image, y in dataloader:
        speed, image, y = [item.to(device).float() for item in [speed, image, y]]
        with torch.no_grad():
            predicted = model(image, speed).float().to(device)
        if len(t_test_predicted_list) <= 0:
            t_test_predicted_list = predicted
            t_test_ground_list = y
        else:
            t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
            t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)

    t_test_predicted_list = test_scale.inverse_transform(t_test_predicted_list)
    t_test_ground_list = test_scale.inverse_transform(t_test_ground_list)
    val_cc = computecc(t_test_predicted_list, t_test_ground_list)
    val_rmse = rmse(t_test_predicted_list, t_test_ground_list)
    val_mae = mae(t_test_predicted_list, t_test_ground_list)
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()

    return test_ground_list, test_predicted_list, val_rmse, val_mae, val_cc
