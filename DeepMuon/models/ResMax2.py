'''
Author: airscker
Date: 2022-12-26 21:36:52
LastEditors: airscker
LastEditTime: 2023-03-06 12:48:21
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit


class ResMax2(nn.Module):
    ''''''
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [5, 4, 3, 2]
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool3d(x)
                                   for x in self.output_num])
        self.conv = nn.Sequential(
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8, 8, 30)),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6, 6, 20)),
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
        self.output_num = [5,4,3]
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool3d(x)
                                   for x in self.output_num])
        self.hit_conv = nn.Sequential(
            ResidualUnit(spatial_dims=3, in_channels=1, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.AdaptiveAvgPool3d((8, 8, 30)),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.AdaptiveAvgPool3d((6, 6, 20)),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
        )
        self.mat_conv = nn.Sequential(
            ResidualUnit(spatial_dims=3, in_channels=2, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.AdaptiveAvgPool3d((8, 8, 30)),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.AdaptiveAvgPool3d((6, 6, 20)),
            ResidualUnit(spatial_dims=3, in_channels=3, out_channels=3, kernel_size=9,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
        )
        self.hidden_size = [512, 128]
        self.flat=nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(1296,3)
            nn.Linear(1296, self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 3),
        )

    def forward(self, x:torch.Tensor):
        batch = x.shape[0]
        hit_num = x[:, 0, :, :, :].unsqueeze(1)
        mat = x[:, 1:, :, :, :]
        hit_feature=self.hit_conv(hit_num)
        mat_feature=self.mat_conv(mat)
        all_feature=torch.cat([mat_feature,hit_feature],dim=1)
        for i in range(len(self.pools)):
            if i == 0:
                feature = self.pools[i](all_feature).view(batch, -1)
            else:
                feature = torch.cat(
                    (feature, self.pools[i](all_feature).view(batch, -1)), 1)
        feature = self.linear_relu_stack(self.flat(feature))
        return F.normalize(feature)
    def freeze_stages(self):
        self.hit_conv.eval()
        self.mat_conv.eval()
        for params in self.hit_conv.parameters():
            params.requires_grad=False
        for params in self.mat_conv.parameters():
            params.requires_grad=False
    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_stages()