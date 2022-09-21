'''
Author: airscker
Date: 2022-09-20 19:35:04
LastEditors: airscker
LastEditTime: 2022-09-20 21:04:22
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
from torch import nn
import torch
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


class MLP3_3D_Fusion(nn.Module):
    def __init__(self):
        '''
        ## Fusion Model Considering the Invade Positon and Invade Direction. Built for 1TeV Hailing data
        - Input: [N,10,10,40,3]
        - Output: [N,6]

        N is the batch size
        '''
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10*40*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,6)
        )
    def forward(self, x):
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

class MLP3_3D_Pos(nn.Module):
    def __init__(self):
        '''
        ## Model Considering the Invade Positon. Built for 1TeV Hailing data
        - Input: [N,10,10,40,3]
        - Output: [N,3]
            
        N is the batch size
        '''
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10*40*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3)
        )
    def forward(self, x):
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

class MLP3_3D_Direc(nn.Module):
    def __init__(self):
        '''
        ## Model Considering the Invade Direction. Built for 1TeV Hailing data
        - Input: [N,10,10,40,3]
        - Output: [N,3]
            
        N is the batch size, and the output direction vector is normalized to 1
        '''
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10*40*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3),
            HailingDirectNorm()
        )
    def forward(self, x):
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

class HailingDirectNorm(nn.Module):
    def __init__(self) -> None:
        '''
        ## Customized Layer, Normalize the Direction Vector of Hailing Data Derived from _Direct Models
        - Input: [N,3], info: [px,py,pz]
        - Output: [N,3], info: [px,py,pz](Normalized)

        N is the batch size, and the output direction vector is normalized to 1
        '''
        super().__init__()
    def forward(self,x):
        return F.normalize(x)

# data=MLP3_3D_Fusion()

