'''
Author: airscker
Date: 2022-09-20 19:33:01
LastEditors: airscker
LastEditTime: 2022-12-27 17:58:52
Description: 
## Multilayer Perceptron Built for Pandax4T III 17*17 Converted Pattern Data
### Corresponding Dataset: `DeepMuon.dataset.Pandax4TData.PandaxDataset`
### Overall Model GFLOPs: 216.32 KMac, Params: 215.68 k

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
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
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
