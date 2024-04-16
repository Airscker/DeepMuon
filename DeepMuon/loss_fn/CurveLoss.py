'''
Author: airscker
Date: 2023-09-16 22:08:04
LastEditors: airscker
LastEditTime: 2024-02-06 23:43:40
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn

class RelativeLoss(nn.Module):
    def __init__(self,integ_ratio:float=1,pos_ratio:float=1,sharp_ratio:float=1,smooth_ratio:float=1) -> None:
        super().__init__()
        self.integ_ratio=integ_ratio
        self.pos_ratio=pos_ratio
        self.sharp_ratio=sharp_ratio
        self.smooth_ratio=smooth_ratio
    def relative_error(self,pred,label):
        curve_range=(torch.max(label,dim=1)[0]-torch.min(label,dim=1)[0])**2
        relative_error=torch.sum((pred-label)**2,dim=1)
        return torch.mean(relative_error/curve_range)
    def integral_error(self,pred,label):
        return torch.square(torch.sum(pred-label))
    def forward(self,pred,label):
        # Integral loss
        integ_loss=self.integral_error(pred,label)
        # Positional loss
        pos_loss=self.relative_error(pred,label)
        # Sharpness loss
        sharpness_loss=self.relative_error(torch.diff(pred,dim=1),torch.diff(label,dim=1))
        # Smoothness loss
        smoothness_loss=self.relative_error(torch.diff(torch.diff(pred,dim=1),dim=1),torch.diff(torch.diff(label,dim=1),dim=1))
        return integ_loss*self.integ_ratio+pos_loss*self.pos_ratio+sharpness_loss*self.sharp_ratio+smoothness_loss*self.smooth_ratio
        