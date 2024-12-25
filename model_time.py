import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import math
import scipy.sparse as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import random


class ModelB(nn.Module):
    def __init__(self, args):
        """一维CNN+Transformerencoder模型结构

        Arguments:
        ----------
            input_dim {int} -- 输入特征的维度
            hidden_dim {int} -- 隐藏层单元数
            Timewindow {int} -- 序列长度


        """
        super(ModelB, self).__init__()
        self.Ck = args.input_length * args.interval
        self.P = args.input_length
        self.b_s = args.batch
        self.kernel_size = args.kernel_size

        self.outchannel = args.outchannel
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.p = args.drop_rate

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=self.outchannel, kernel_size=self.kernel_size)  # 实际上卷积大小为kernel_size*in_channels
        self.LSTM = nn.LSTM(input_size=self.outchannel, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.BiLSTM = nn.LSTM(input_size=self.outchannel, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bidirectional=True)

        self.dense = nn.Linear(self.hidden_size * 2, 100)
        self.dropout = nn.Dropout(p=self.p)
        linear_input = (args.input_length * 24 - self.kernel_size + 1) * 100
        self.fc = nn.Linear(linear_input, 100)

    def forward(self, input):
        # ipt是输入序列
        # print("input.shape: ", input.shape)  # torch.Size([32, 24, 6])
        input = torch.transpose(input, 2, 1)
        ipt = self.conv1(input)                                 # torch.Size([32, 32, 22])  (batch_size, input_size, seq_length)
        ipt = ipt.permute(2, 0, 1)                              # torch.Size([22, 32, 32])  ([24,32,34])
        ipt_output, (hn, cn) = self.BiLSTM(ipt)                 # torch.Size([22, 32, 200]) 
        ipt_output = ipt_output.permute(1, 0, 2)                # torch.Size([32, 22, 200])
        ipt_output = self.dense(ipt_output)                     # torch.Size([32, 22, 100])
        ipt_output = self.dropout(ipt_output)                   # torch.Size([32, 22, 100])
        ipt_output = ipt_output.view(ipt_output.size(0), -1)    # torch.Size([32, 2200])
        ipt_output = self.fc(ipt_output)                        # torch.Size([32, 100])
        return ipt_output
