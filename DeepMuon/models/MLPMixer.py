'''
Author: airscker
Date: 2023-04-30 15:39:26
LastEditors: airscker
LastEditTime: 2023-05-15 23:56:04
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn

class MLPBlock(nn.Module):
    def __init__(self, dim:int, hidden_size:int, dropout=0.1) -> None:
        super().__init__()
        self.linear=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,dim)
        )
    def forward(self,x):
        return self.linear(x)

class MixerBlock(nn.Module):
    def __init__(self,dim:int,channel:int,token_drop=0.1,channel_drop=0.1) -> None:
        super().__init__()
        self.token_mixer=MLPBlock(dim=dim,hidden_size=channel,dropout=token_drop)
        self.channel_mixer=MLPBlock(dim=channel,hidden_size=dim,dropout=channel_drop)
    def forward(self,x):
        x=x+self.token_mixer(x)
        x=torch.permute(x,(0,2,1))
        x=x+self.channel_mixer(x)
        x=torch.permute(x,(0,2,1))
        return x

class MLPMixer(nn.Module):
    """
    ## A PyTorch implementation of the MLP-Mixer architecture for image classification.

    ### Args:
        - depth (int): Number of MixerBlocks in the model.
        - dim (int): Hidden dimension of the MixerBlocks.
        - channel (int): Number of channels in the input image.
        - token_drop (float): Dropout rate for the token mixing MLP.
        - channel_drop (float): Dropout rate for the channel mixing MLP.
        - classes (int): Number of output classes.

    ### Inputs:
        - x (torch.Tensor): Input tensor of shape (N, dim, channel).

    ### Outputs:
        - logits (torch.Tensor): Output tensor of shape (N, classes).
    """
    def __init__(self,
                 depth: int = 1,
                 dim: int = 100,
                 channel: int = 2,
                 token_drop: float = 0.1,
                 channel_drop: float = 0.1,
                 classes: int = 3):
        super().__init__()
        self.mixers = nn.Sequential(
            *[MixerBlock(dim=dim, channel=channel, token_drop=token_drop, channel_drop=channel_drop) for _ in range(depth)]
            )
        self.linear=nn.Linear(dim,classes)
        self.reset_parameters()
    def reset_parameters(self):
        for name,para in self.named_parameters():
            if name=='weight':
                nn.init.kaiming_uniform_(para)
            elif name=='bias':
                nn.init.zeros_(para)
    def forward(self,x:torch.Tensor):
        x=self.mixers(x)
        x=torch.mean(x,dim=1)
        x=self.linear(x)
        return x
    
class XASMLP(nn.Module):
    def __init__(self,input_node=100,classes=3,dropout=0.1):
        super().__init__()
        self.hidden_nodes=[256,128]
        self.linear=nn.Sequential(
            nn.Linear(input_node,self.hidden_nodes[0]),
            nn.BatchNorm1d(self.hidden_nodes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[0],self.hidden_nodes[1]),
            nn.BatchNorm1d(self.hidden_nodes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_nodes[1],classes)
        )
    def forward(self,x):
        return self.linear(x)