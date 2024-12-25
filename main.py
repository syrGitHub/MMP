# coding=UTF-8
import os
from process import process
import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, random_split, Subset
import argparse
from data import DataGenerator

from train import train
from test import val, test
from env import *

import random
from datetime import datetime
from pathlib import Path
from model import Model


class Main():
    def __init__(self, args):
        self.args = args
        self.datestr = None
        set_device(self.args.device)
        self.device = get_device()

        self.train_dataset = DataGenerator(self.args, 'train', self.args.interval)
        self.val_dataset = DataGenerator(self.args, 'val', self.args.interval)
        self.test_dataset = DataGenerator(self.args, 'test', self.args.interval)

        self.train_scale_y = self.train_dataset.scale_y
        self.val_scale_y = self.val_dataset.scale_y
        self.test_scale_y = self.test_dataset.scale_y

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                           drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                         drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                          drop_last=True)

        self.model = Model(args).to(self.device)

        # print(list(self.model.parameters()))
        # pretrained_dict = self.model.state_dict()
        # 打印权重信息
        # print(pretrained_dict.items())

    def run(self, rang):

        if len(self.args.load_model_path) > 0:
            model_save_path = self.args.load_model_path
            print("main_model_save_path_load_model_path:", model_save_path)
        else:
            model_save_path = self.get_save_path(rang)[0]
            print("main_model_save_path:", model_save_path)
            print(self.model)
            nParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # nParams = sum([p.nelement() for p in self.model.parameters()])
            print('Number of model parameters is', nParams)
            self.train_log = train(self.model, model_save_path,
                                   args=self.args,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=self.val_dataloader,
                                   train_scale_y=self.train_scale_y,
                                   val_scale_y=self.val_scale_y
                                   )
            # print("main_self.train_log:", self.train_log)

        # test
        # self.model.load_state_dict(torch.load(model_save_path))
        self.model = torch.load(model_save_path)
        print("load ok")
        best_model = self.model.to(self.device)
        nParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # nParams = sum([p.nelement() for p in self.model.parameters()])
        print('Number of test model parameters is', nParams)

        train_true_result, train_result, train_rmse, train_mae, train_cc = test(best_model, self.train_dataloader,
                                                                                self.train_scale_y)
        val_true_result, val_result, val_rmse, val_mae, val_cc = test(best_model, self.val_dataloader,
                                                                      self.val_scale_y)
        test_true_result, test_result, test_rmse, test_mae, test_cc = test(best_model, self.test_dataloader,
                                                                           self.test_scale_y)

        print("Prediction Result:", train_rmse.item(), train_mae.item(), train_cc.item(), val_rmse.item(), val_mae.item(), val_cc.item(),
              test_rmse.item(), test_mae.item(), test_cc.item())
        return train_rmse.item(), train_mae.item(), train_cc.item(), val_rmse.item(), val_mae.item(), val_cc.item(),\
               test_rmse.item(), test_mae.item(), test_cc.item()

    def get_save_path(self, rang):

        dir_path = self.args.save_path_pattern

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./Test_results/{dir_path}/best_{datestr}_{rang}.pt',
            f'./Test_results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='batch size', type=int, default=32)
    parser.add_argument('--epoch', help='train epoch', type=int, default=30)
    parser.add_argument('--save_path_pattern', help='save path pattern', type=str, default='Test')
    parser.add_argument('--load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('--device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('--random_seed', help='random seed', type=int, default=5)
    parser.add_argument('--report', help='best / val', type=str, default='best')
    parser.add_argument('--interval', help='EUV数据采样频率（间隔几张采一次）', type=int, default=24)
    parser.add_argument('--input_length', help='input_length', type=int, default=1)  # 输入数据长度
    parser.add_argument('--predict_length', help='predict length', type=int, default=1)  # 输出数据长度
    parser.add_argument('--norm', help='normalize type', type=int, default=3)  # 正则化方式


    # parser.add_argument('--hidCNN', type=int, default=2048, help='number of CNN hidden units')
    parser.add_argument('--CNN_kernel', type=int, default=3, help='the kernel size of the CNN layers')  # 卷积核大小
    parser.add_argument('--decay', help='decay', type=float, default=0)
    parser.add_argument('--outchannel', type=int, default=32, help='out channel applied to conv1d')
    parser.add_argument('--num_layers', type=int, default=1, help='layers of LSTM')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden size of LSTM')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of conv1d')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')

    args = parser.parse_args()
    print("训练参数： ", args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_rmse, train_mae, train_cc, val_rmse, val_mae, val_cc, test_rmse, test_mae, test_cc = [], [], [], [], [], [],\
                                                                                               [], [], []
    if len(args.load_model_path) > 0:
        main = Main(args)
        train_rmse_1, train_mae_1, train_cc_1, val_rmse_1, val_mae_1, val_cc_1, test_rmse_1, test_mae_1, test_cc_1 = \
            main.run(0)
        train_rmse.append(train_rmse_1)
        train_mae.append(train_mae_1)
        train_cc.append(train_cc_1)
        val_rmse.append(val_rmse_1)
        val_mae.append(val_mae_1)
        val_cc.append(val_cc_1)
        test_rmse.append(test_rmse_1)
        test_mae.append(test_mae_1)
        test_cc.append(test_cc_1)
        print(train_rmse, train_mae, train_cc, val_rmse, val_mae, val_cc, test_rmse, test_mae, test_cc)
    else:
        for i in range(1):
            print("第几轮：", i)
            main = Main(args)
            train_rmse_1, train_mae_1, train_cc_1, val_rmse_1, val_mae_1, val_cc_1, test_rmse_1, test_mae_1, test_cc_1 = \
                main.run(i)
            train_rmse.append(train_rmse_1)
            train_mae.append(train_mae_1)
            train_cc.append(train_cc_1)
            val_rmse.append(val_rmse_1)
            val_mae.append(val_mae_1)
            val_cc.append(val_cc_1)
            test_rmse.append(test_rmse_1)
            test_mae.append(test_mae_1)
            test_cc.append(test_cc_1)
        print(train_rmse, train_mae, train_cc, val_rmse, val_mae, val_cc, test_rmse, test_mae, test_cc)
        print(np.mean(train_rmse), np.std(train_rmse, ddof=1), np.mean(train_mae), np.std(train_mae, ddof=1),
              np.mean(train_cc), np.std(train_cc, ddof=1),
              np.mean(val_rmse), np.std(val_rmse, ddof=1), np.mean(val_mae), np.std(val_mae, ddof=1),
              np.mean(val_cc), np.std(val_cc, ddof=1),
              np.mean(test_rmse), np.std(test_rmse, ddof=1), np.mean(test_mae), np.std(test_mae, ddof=1),
              np.mean(test_cc), np.std(test_cc, ddof=1))
        print(np.min(val_rmse))


