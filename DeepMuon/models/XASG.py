'''
Author: airscker
Date: 2023-07-11 08:17:41
LastEditors: airscker
LastEditTime: 2023-07-13 12:25:21
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import torch
from torch import nn

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