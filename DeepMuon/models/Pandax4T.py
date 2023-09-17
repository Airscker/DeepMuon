'''
Author: airscker
Date: 2022-09-20 19:33:01
LastEditors: airscker
LastEditTime: 2023-09-13 21:06:13
Description: 
## Multilayer Perceptron Built for Pandax4T III 17*17 Converted Pattern Data
### Corresponding Dataset: `DeepMuon.dataset.Pandax4TData.PandaxDataset`
### Overall Model GFLOPs: 216.32 KMac, Params: 215.68 k

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

from torch import nn
from .base import MLPBlock
import torch


class MLP3(nn.Module):
    def __init__(self,in_dim=17*17,n_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack=MLPBlock(in_dim,n_classes,[512,512],normalization=nn.BatchNorm1d,activation=nn.LeakyReLU)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
