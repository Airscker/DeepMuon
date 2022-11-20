'''
Author: airscker
Date: 2022-10-11 19:40:55
LastEditors: airscker
LastEditTime: 2022-10-12 00:05:39
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import *
torch.set_default_tensor_type(torch.DoubleTensor)

class LinearRegression3D(nn.Module):
    def __init__(self,layers=40) -> None:
        super().__init__()
        # input data shape should be [N,layers,10,10,3]
        self.layer_cluster=nn.Sequential(
            nn.Conv3d(layers,layers,(5,5,1),groups=layers),
            nn.BatchNorm3d(layers),
            nn.Tanh()
        )
        # self.layer_pos=nn.parameter(torch.tensor([]))
        # self.Layer_cluster=nn.ModuleList(
        #     [SingleLayer_cluster for x in range(len(layers))]
        # )
        # self.vit=ViT(3,[10,10,layers],[10,10,layers],hidden_size=3,num_layers=12,num_heads=3,mlp_dim=1024)
        self.linear_regression=nn.Sequential(
            nn.Flatten(),
        #     # nn.Linear(3,3),
        #     nn.Linear(10*10*layers*3,2048),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(),
        #     nn.Linear(2048,512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512,3)
            nn.Linear(layers*36,layers),
            nn.BatchNorm1d(layers),
            nn.LeakyReLU(),
            nn.Linear(layers,3),
            HailingDirectNorm()
        )
    def forward(self,x):
        # input shape [N,3,10,10,layers]
        x=self.layer_cluster(x.permute(0,4,2,3,1))
        x=self.linear_regression(x)
        # for layer in self.Layer_cluster:
        #     pos=layer(x[:,:,:,])
        return x

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

class SingleLayer_cluster(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):   
        x=self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

