'''
Author: airscker
Date: 2022-09-21 23:59:56
LastEditors: airscker
LastEditTime: 2022-09-22 00:06:31
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
'''
CNN
use Conv3d  MaxPool3d   Linear  LeakyReLU  
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=8,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=8,out_channels=24,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=24,out_channels=64,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64,out_channels=128,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,3)
        )
    def forward(self, x):
        x=self.conv(x)
        x=self.flatten(x)
        x=self.linear_relu_stack(x)
        return 100*F.normalize(x)


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=8,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32,out_channels=64,kernel_size=(3,3,10),stride=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,3)
        )
    def forward(self, x):
        x=self.conv(x)
        x=self.flatten(x)
        x=self.linear_relu_stack(x)
        return 100*F.normalize(x)
