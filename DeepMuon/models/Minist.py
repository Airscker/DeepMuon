'''
Author: airscker
Date: 2023-05-15 13:40:07
LastEditors: airscker
LastEditTime: 2023-09-13 21:06:16
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

from torch import nn
from .base import MLPBlock

class MinistModel(nn.Module):
    def __init__(self,img_size=28,n_classes=10,hidden_size=[512,512]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = MLPBlock(img_size**2,n_classes,hidden_size)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits