'''
Author: airscker
Date: 2022-09-20 19:43:46
LastEditors: airscker
LastEditTime: 2022-10-10 19:52:14
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class MSALoss(nn.Module):
    def __init__(self,angle_ratio=1):
        """## MSEloss(vec1,vec2)+Angle(vec1,vec2)
        - Args:
            - angle_ratio (int, optional): The ratio to consider the angle loss into total loss. Defaults to 1.
        """
        super().__init__()
        self.angle_ratio=angle_ratio
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(input)
        mseloss=(input-target)**2
        mseloss=torch.sum(mseloss)/(mseloss.shape[0]*mseloss.shape[1])
        # angloss=ang(input=input,target=target)``
        angloss=angn(input=input,target=target)
        return mseloss+self.angle_ratio*angloss

def ang(input,target):
    res=torch.zeros(input.shape[0])
    for i in range(input.shape[0]):
        res[i]=torch.dot(input[i],target[i])/(torch.sqrt(torch.sum(input[i]**2)*torch.sum(target[i]**2)))
    res=torch.mean(torch.arccos(res))
    return res

def angn(input,target):
    input=input.detach().cpu().numpy()
    target=target.detach().cpu().numpy()
    res=np.zeros(input.shape[0])
    for i in range(input.shape[0]):
        res[i]=np.dot(input[i],target[i])/(np.sqrt(np.sum(input[i]**2)*np.sum(target[i]**2)))
    res=np.mean(np.arccos(res))
    return torch.from_numpy(np.array(res))
        