'''
Author: airscker
Date: 2023-03-14 16:57:11
LastEditors: airscker
LastEditTime: 2023-03-14 16:57:28
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F

class CLS_REC_KLD(nn.Module):
    def __init__(self,a=1,b=0.6,c=0.4) -> None:
        super().__init__()
        self.a=a
        self.b=b
        self.c=c
    def forward(self,data,cla_out,rec_out,fea,fea2,target):
        cls_loss=F.cross_entropy(cla_out,target)
        rec_loss=F.mse_loss(rec_out,data)
        con_loss=F.kl_div(fea,fea2)
        return self.a*cls_loss+self.b*rec_loss+self.c*con_loss