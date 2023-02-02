'''
Author: airscker
Date: 2023-01-22 09:25:11
LastEditors: airscker
LastEditTime: 2023-01-22 20:54:02
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


class TransConv(nn.Module):
    def __init__(self, patch_size=(2, 2, 8), mlp_drop_rate=0, poolsize=(5, 5, 5)) -> None:
        super().__init__()
        embed_output_channel = 3*np.prod(patch_size)
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=embed_output_channel,
                      kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm3d(embed_output_channel),
            nn.LeakyReLU()
        )
        self.depwise_conv = nn.Sequential(
            nn.Conv3d(embed_output_channel, embed_output_channel,
                      kernel_size=3, groups=embed_output_channel, padding=1),
            nn.BatchNorm3d(embed_output_channel),
            nn.LeakyReLU()
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(embed_output_channel,
                      embed_output_channel, kernel_size=1),
            nn.BatchNorm3d(embed_output_channel),
            nn.LeakyReLU()
        )
        self.pool = nn.AdaptiveAvgPool3d(poolsize)
        self.hidden_size = [512, 256]
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_output_channel *
                      np.prod(poolsize), self.hidden_size[0]),
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
        patch_embed_out = self.depwise_conv(self.patch_embed(x))
        res = self.depwise_conv(patch_embed_out)
        feature = patch_embed_out+res
        feature = self.pointwise_conv(feature)
        feature = self.pool(feature)
        return F.normalize(self.linear_relu_stack(feature))
