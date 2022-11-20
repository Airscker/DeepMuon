'''
Author: airscker
Date: 2022-09-20 19:33:01
LastEditors: airscker
LastEditTime: 2022-09-21 23:10:42
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

from torch import nn
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17*17, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):   
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

class MLP3v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17*17, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):   
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

class CONV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10,out_channels=40,kernel_size=(5,5)),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=40,out_channels=80,kernel_size=(5,5)),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):   
        x=self.conv(x)
        x=self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
