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
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = MLPBlock(28*28,10,[512,512])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits