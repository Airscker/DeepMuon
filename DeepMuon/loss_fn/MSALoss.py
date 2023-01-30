'''
Author: airscker
Date: 2022-09-20 19:43:46
LastEditors: airscker
LastEditTime: 2023-01-30 22:11:31
Description: NULL

Copyright (C) 2022 by Airscker, All Rights Reserved. 
'''

import torch
from torch import nn
from torch import Tensor


class MSALoss(nn.Module):
    def __init__(self, angle_ratio=1, len_ratio=0):
        """## MSEloss(vec1,vec2)+Angle(vec1,vec2)
        - Args:
            - angle_ratio (int, optional): The ratio to consider the angle loss into total loss. Defaults to 1.
        """
        super().__init__()
        self.angle_ratio = angle_ratio
        self.len_ratio = len_ratio

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(input)
        # mseloss=torch.mean((input-target)**2)
        angloss = torch.mean(torch.sin(torch.arccos(torch.sum(
            input*target, axis=1)/torch.sqrt(torch.sum(input**2, axis=1)*torch.sum(target**2, axis=1)))))
        # return self.len_ratio*mseloss+self.angle_ratio*angloss
        return angloss


class MSALoss2(nn.Module):
    def __init__(self, angle_ratio=1, len_ratio=0):
        """## MSEloss(vec1,vec2)+Angle(vec1,vec2)
        - Args:
            - angle_ratio (int, optional): The ratio to consider the angle loss into total loss. Defaults to 1.
        """
        super().__init__()
        self.angle_ratio = angle_ratio
        self.len_ratio = len_ratio

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(input)
        # mseloss=torch.mean((input-target)**2)
        angloss = torch.mean(1-(torch.sum(input*target, axis=1)/torch.sqrt(
            torch.sum(input**2, axis=1)*torch.sum(target**2, axis=1)))**2)
        # return self.len_ratio*mseloss+self.angle_ratio*angloss
        return angloss
