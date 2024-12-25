# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

import math
from model_image import ModelA
from model_time import ModelB


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.ModelA = ModelA(args)    # image model
        self.ModelB = ModelB(args)    # timeline model
        
        self.Linear_layer = nn.Linear(1100, 1)

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, image, speed):
        # print(image.shape, speed.shape)   # torch.Size([32, 3, 256, 256]) torch.Size([32, 6, 3])
        x = self.ModelA(image)              # torch.Size([32, 4096])
        y = self.ModelB(speed)              # torch.Size([32, 100])

        out = torch.cat((x, y), 1)          # torch.Size([32, 4196])
        out = out.reshape(out.size(0), -1)  # torch.Size([32, 4196])
        out = self.Linear_layer(out)        # torch.Size([32, 1])

        return out

