'''
Author: airscker
Date: 2022-10-04 02:10:18
LastEditors: airscker
LastEditTime: 2022-10-05 11:21:11
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
from monai.networks.nets.unet import UNet
from monai.networks.nets.vit import ViT
import torch
from torch import nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


class UNET_MLP(nn.Module):
    def __init__(self,mlp_drop_rate=0):
        super().__init__()
        self.unet=UNet(spatial_dims=3,in_channels=3,out_channels=1,channels=(6,12,24),strides=(1,1,1),num_res_units=3)
        self.vit=ViT(1,[10,10,40],[10,10,10],hidden_size=32,num_layers=3,num_heads=16,mlp_dim=32)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(4*32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop_rate),
            nn.Linear(128, 3),
            HailingDirectNorm()
        )
    def forward(self,x):
        x,_=self.vit(self.unet(x))
        return self.mlp(self.flatten(x))

class UNET_MLP_D(nn.Module):
    def __init__(self,mlp_drop_rate=0):
        super().__init__()
        self.unet=UNet(spatial_dims=3,in_channels=3,out_channels=1,channels=(6,12,24),strides=(1,1,1),num_res_units=3)
        # self.vit=ViT(1,[10,10,40],[10,10,10],hidden_size=32,num_layers=3,num_heads=16,mlp_dim=32)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(10*10*40, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop_rate),
            nn.Linear(128, 3),
            HailingDirectNorm()
        )
    def forward(self,x):
        x=self.unet(x)
        return self.mlp(self.flatten(x))

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