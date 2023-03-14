'''
Author: airscker
Date: 2022-12-26 21:36:52
LastEditors: airscker
LastEditTime: 2023-03-13 14:12:03
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from monai.networks.blocks.convolutions import ResidualUnit


class ResMax_C1(nn.Module):
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [32, 16, 8]
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool3d(x)
                                   for x in self.output_num])
        self.conv1 = nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=3, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[0]))
        self.conv2=nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=6, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[1]))
        self.conv3=nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=6, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[2]))
        self.hidden_size = [1024, 256]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(int(np.sum(np.array(self.output_num)**2)*6), self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 2),
            nn.Softmax(1)
        )

    def freeze_stages(self):
        pass

    def forward(self, x:torch.Tensor):
        batch=x.shape[0]
        conv_f1=self.conv1(x)
        conv_f2=self.conv2(conv_f1)
        conv_f3=self.conv3(conv_f2)
        linear_feature=torch.cat([conv_f1.view(batch,-1),conv_f2.view(batch,-1),conv_f3.view(batch,-1)],1)
        linear_feature=self.linear_relu_stack(linear_feature)
        return linear_feature
    
class ResNet50_C1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=resnet50()
        self.model.fc=nn.Linear(self.model.fc.in_features,2)
        self.softmax=nn.Softmax(1)
    def forward(self,x):
        return self.softmax(self.model(x))