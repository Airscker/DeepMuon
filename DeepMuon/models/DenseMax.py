'''
Author: airscker
Date: 2022-12-26 21:36:52
LastEditors: airscker
LastEditTime: 2023-01-22 19:01:49
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit
torch.set_default_tensor_type(torch.DoubleTensor)


class DenseMax(nn.Module):
    def __init__(self, dense_length=3, mlp_drop_rate=0) -> None:
        super().__init__()
        self.output_num = [5, 4, 3, 2]
        self.max_pools = nn.ModuleList([nn.AdaptiveMaxPool3d(x)
                                        for x in self.output_num])
        self.hit_dense_list = nn.ModuleList(
            [ResidualUnit(spatial_dims=3, in_channels=1,
                          out_channels=1, kernel_size=5, subunits=2, dropout=0) for x in range(dense_length)]
        )
        self.mat_dense_list = nn.ModuleList(
            [ResidualUnit(spatial_dims=3, in_channels=2,
                          out_channels=2, kernel_size=5, subunits=2, dropout=0) for x in range(dense_length)]
        )
        self.conv = nn.Sequential(
            nn.Conv3d(3, 3, 3),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
        )
        self.flat = nn.Flatten()
        self.hidden_size = [512, 128]
        self.mlp = nn.Sequential(
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

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        hit_num = x[:, 0, :, :, :].unsqueeze(1)
        mat = x[:, 1:, :, :, :]
        hit_num_dense_out = []
        for layer in self.hit_dense_list:
            hit_num_dense_out.append(layer(hit_num))
            for item in hit_num_dense_out:
                hit_num += item
        mat_dense_out = []
        for layer in self.mat_dense_list:
            mat_dense_out.append(layer(mat))
            for item in mat_dense_out:
                mat += item
        concat_output = self.conv(torch.cat((mat, hit_num), 1))
        for i in range(len(self.max_pools)):
            if i == 0:
                feature = self.max_pools[i](concat_output).view(batch, -1)
            else:
                feature = torch.cat(
                    (feature, self.max_pools[i](concat_output).view(batch, -1)), 1)
        feature = self.mlp(self.flat(feature))
        return feature
