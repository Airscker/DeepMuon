'''
Author: airscker
Date: 2023-04-30 15:39:26
LastEditors: airscker
LastEditTime: 2023-04-30 15:57:34
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn
import numpy as np

class ValenceV1(nn.Module):
    def __init__(self,input_node=100,classes=10) -> None:
        super().__init__()
        self.nodes=[256,64]
        self.linear=nn.Sequential(
            nn.Linear(input_node,self.nodes[0]),
            nn.ReLU(),
            nn.Linear(self.nodes[0],self.nodes[1]),
            nn.ReLU(),
            nn.Linear(self.nodes[1],classes)
            )
    def forward(self,x:torch.Tensor):
        x=self.linear(x)
        return x