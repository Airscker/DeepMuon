'''
Author: airscker
Date: 2023-01-23 17:10:49
LastEditors: airscker
LastEditTime: 2023-01-23 17:30:04
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


class FCN1(nn.Module):
    def __init__(self, drop_out=0.2) -> None:
        super().__init__()
        hidden_size = [2400, 1200, 300, 100]
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(3*10*10*40, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.BatchNorm1d(hidden_size[2]),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size[2], hidden_size[3]),
            nn.BatchNorm1d(hidden_size[3]),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size[3], 3)
        )

    def forward(self, x):
        return F.normalize(self.mlp(self.flat(x)))
