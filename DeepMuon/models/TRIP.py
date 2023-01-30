'''
Author: airscker
Date: 2022-12-04 22:57:15
LastEditors: airscker
LastEditTime: 2023-01-21 10:52:27
Description: Trilateral Projection Neural Network
    - Input
        - shape: [N,3,10,10,40/50]
    - Output
        - note: normalized three dimensional direction vector

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit


class BotP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ResidualUnit(2, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            ResidualUnit(2, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        '''
        Input shape: [N,3,10,10]
        Output shape: [N,3]
        '''
        return self.mlp(self.conv(x))


class SideP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ResidualUnit(2, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            ResidualUnit(2, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU()
        )
        self.down_sample = nn.AdaptiveAvgPool2d((10, 10))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        '''
        Input shape: [N,3,10,40]
        Output shape: [N,3]
        '''
        return self.mlp(self.down_sample(self.conv(x)))


class TRIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projects = nn.ModuleList([nn.AdaptiveAvgPool3d((10, 10, 1)),
                                       nn.AdaptiveAvgPool3d((10, 1, 40)),
                                       nn.AdaptiveAvgPool3d((1, 10, 40))])
        self.regs = nn.ModuleList([BotP(), SideP(), SideP()])
        self.vec = nn.Sequential(
            nn.Linear(3, 9),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(),
            nn.Linear(9, 1)
        )

    def forward(self, x: torch.Tensor):
        xyp = self.regs[0](self.projects[0](x).squeeze(-1)).unsqueeze(1)
        xzp = self.regs[1](self.projects[1](x).squeeze(-2)).unsqueeze(1)
        yzp = self.regs[2](self.projects[2](x).squeeze(-3)).unsqueeze(1)
        x = self.vec(torch.cat([xyp, xzp, yzp], 1)).squeeze(-1)
        return F.normalize(x)
