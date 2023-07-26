'''
Author: airscker
Date: 2023-07-11 08:17:41
LastEditors: airscker
LastEditTime: 2023-07-26 12:50:39
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import math
import torch
from torch import nn
from monai.networks.blocks.convolutions import ResidualUnit

class MLPBlock(nn.Module):
    def __init__(self,input_node=100,classes=3,dropout=0.1):
        super().__init__()
        self.hidden_nodes=[5120,2048,1024,512]
        self.linear=nn.Sequential(
            nn.Linear(input_node,self.hidden_nodes[0]),
            nn.BatchNorm1d(self.hidden_nodes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[0],self.hidden_nodes[1]),
            nn.BatchNorm1d(self.hidden_nodes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[1],self.hidden_nodes[2]),
            nn.BatchNorm1d(self.hidden_nodes[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[2],self.hidden_nodes[3]),
            nn.BatchNorm1d(self.hidden_nodes[3]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[3],classes)
        )
    def forward(self,x):
        return self.linear(x)

class XASGV1(nn.Module):
    def __init__(self, mlp_pretrained=None,mlp_drop_out=0.1) -> None:
        super().__init__()
        self.mlp1=MLPBlock(input_node=100,classes=1,dropout=mlp_drop_out)
        if os.path.exists(mlp_pretrained):
            pretrained=torch.load(mlp_pretrained)['model']
            try:
                self.mlp1.load_state_dict(pretrained)
            except:
                pass
            # for name,para in self.mlp1.named_parameters():
            #     if 'linear.8' not in name:
            #         para.requires_grad=False

    def forward(self,x):
        return self.mlp1(x)

class XASGV2(nn.Module):
    def __init__(self, input_node=100,mlp_drop_out=0.1,classes=1) -> None:
        super().__init__()
        self.hidden_nodes=[512,256,128,64]
        self.mlp1=nn.Sequential(
            nn.Linear(input_node,self.hidden_nodes[0]),
            nn.BatchNorm1d(self.hidden_nodes[0]),
            nn.ReLU(),
            nn.Dropout(mlp_drop_out),
            nn.Linear(self.hidden_nodes[0],self.hidden_nodes[1]),
            nn.BatchNorm1d(self.hidden_nodes[1]),
            nn.ReLU(),
            nn.Dropout(mlp_drop_out),
            nn.Linear(self.hidden_nodes[1],self.hidden_nodes[2]),
            nn.BatchNorm1d(self.hidden_nodes[2]),
            nn.ReLU(),
            nn.Dropout(mlp_drop_out),
            nn.Linear(self.hidden_nodes[2],self.hidden_nodes[3]),
            nn.BatchNorm1d(self.hidden_nodes[3]),
            nn.ReLU(),
            nn.Dropout(mlp_drop_out),
            nn.Linear(self.hidden_nodes[3],classes)
        )
    def forward(self,x):
        return self.mlp1(x)
    
class TransBlock(nn.Module):
    def __init__(self,x_dim=100,y_dim=2) -> None:
        super().__init__()
        self.qkv=nn.Linear(x_dim,x_dim*3)
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.dim_norm=1/math.sqrt(x_dim)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''x.shape: [B,X,Y]'''
        output=self.qkv(x)
        output=output.view((x.shape[0],self.y_dim,3,-1)).contiguous()
        q,k,v=output[:,:,0,:],output[:,:,1,:],output[:,:,2,:]
        output=torch.softmax(torch.bmm(q,k.permute(0,2,1).contiguous())*self.dim_norm,dim=-1)
        output=torch.bmm(output,v)
        return output
class TransXAS(nn.Module):
    def __init__(self,x_dim=100,y_dim=2,heads=10,hidden_size=[1024,512],out_dim=1) -> None:
        super().__init__()
        self.heads=heads
        self.multi_heads=nn.ModuleList([
            TransBlock(x_dim=x_dim,y_dim=y_dim) for _ in range(heads)
        ])
        self.conv=nn.Sequential(
            ResidualUnit(spatial_dims=1,in_channels=heads*y_dim,out_channels=heads*y_dim),
            nn.LayerNorm([heads*y_dim,x_dim]),
            nn.LeakyReLU(),
            ResidualUnit(spatial_dims=1,in_channels=heads*y_dim,out_channels=heads*y_dim),
            nn.LayerNorm([heads*y_dim,x_dim]),
            nn.LeakyReLU()
        )
        self.flatten=nn.Flatten()
        self.mlp=nn.Sequential(
            nn.Linear(heads*y_dim*2*x_dim,hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[0],hidden_size[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[1],out_dim)
        )
    def forward(self,x:torch.Tensor):
        '''x.shape: [B,X,Y]'''
        for i in range(self.heads):
            if i == 0:
                trans_feature=self.multi_heads[i](x)
            else:
                trans_feature=torch.cat([trans_feature,self.multi_heads[i](x)],dim=1)
        conv_feature=self.conv(trans_feature)
        features=torch.cat([trans_feature,conv_feature],dim=1)
        features=self.mlp(self.flatten(features))
        return features