'''
Author: airscker
Date: 2022-09-20 19:43:46
LastEditors: airscker
LastEditTime: 2022-10-27 17:59:38
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class MSALoss(nn.Module):
    def __init__(self,angle_ratio=1,len_ratio=0):
        """## MSEloss(vec1,vec2)+Angle(vec1,vec2)
        - Args:
            - angle_ratio (int, optional): The ratio to consider the angle loss into total loss. Defaults to 1.
        """
        super().__init__()
        self.angle_ratio=angle_ratio
        self.len_ratio=len_ratio
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(input)
        # mseloss=torch.mean((input-target)**2)
        angloss=torch.mean(torch.sin(torch.arccos(torch.sum(input*target,axis=1)/torch.sqrt(torch.sum(input**2,axis=1)*torch.sum(target**2,axis=1)))))
        # return self.len_ratio*mseloss+self.angle_ratio*angloss
        return angloss