'''
Author: airscker
Date: 2022-12-26 21:36:52
LastEditors: airscker
LastEditTime: 2023-01-27 19:50:01
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit
torch.set_default_tensor_type(torch.DoubleTensor)


class ResMax2(nn.Module):
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [5, 4, 3, 2]
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool3d(x)
                                   for x in self.output_num])
        self.conv = nn.Sequential(
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8, 8, 30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6, 6, 20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
        )
        self.hidden_size = [1024, 256]
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(672, self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 3),
        )

    def freeze_stages(self):
        self.conv.eval()
        for params in self.conv.parameters():
            params.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x = self.conv(x)
        for i in range(len(self.pools)):
            if i == 0:
                feature = self.pools[i](x).view(batch, -1)
            else:
                feature = torch.cat(
                    (feature, self.pools[i](x).view(batch, -1)), 1)
        x = self.linear_relu_stack(feature)
        return F.normalize(x)


class ResMax3(nn.Module):
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [4, 3, 2, 1]
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool3d(x)
                                   for x in self.output_num])
        self.conv = nn.Sequential(
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d((8, 8, 30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d((6, 6, 20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=1, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
        )
        self.hidden_size = [512, 128]
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(300, self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 3),
        )

    def forward(self, x):
        batch = x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x = self.conv(x)
        for i in range(len(self.pools)):
            if i == 0:
                feature = self.pools[i](x).view(batch, -1)
            else:
                feature = torch.cat(
                    (feature, self.pools[i](x).view(batch, -1)), 1)
        x = self.linear_relu_stack(feature)
        return F.normalize(x)
