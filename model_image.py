# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

import math

class ModelA(nn.Module):
    def __init__(self, args):
        super(ModelA, self).__init__()
        inchannel = args.input_length
        print("inchannelinchannel:", inchannel)
        model = models.googlenet(pretrained=True)

        # 修改网络的第一个卷积层的输入为4通道
        print("before:", model)
        print("model.conv1.conv:", model.conv1.conv)
        model.conv1.conv = nn.Conv2d(inchannel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.resnet_layer = model

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):
        # print(data.shape)   # torch.Size([32, 3, 256, 256])
        x = self.resnet_layer(data)
        return x
