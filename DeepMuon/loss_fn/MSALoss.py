'''
Author: airscker
Date: 2023-01-30 21:10:09
LastEditors: airscker
LastEditTime: 2023-02-16 17:59:20
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn
from torch import Tensor


class MSALoss(nn.Module):
    """
    ## MSEloss(vec1,vec2)+Angle(vec1,vec2)

    ### Args:
        - angle_ratio: The ratio to consider the angle loss into total loss. Defaults to 1.
        - len_ratio: The ratio to consider the distance loss into total loss. Defaults to 1.
    """

    def __init__(self, angle_ratio=1, len_ratio=0):

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
