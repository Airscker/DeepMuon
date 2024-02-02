'''
Author: airscker
Date: 2022-12-23 10:33:54
LastEditors: airscker
LastEditTime: 2023-12-11 15:33:16
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import torch
import torch.nn.functional as F
from torch import nn
from ..base import SwinTransformer3D

class VST(nn.Module):
    def __init__(self, mlp_drop=0, **kwargs) -> None:
        super().__init__()
        hiddensize = [512, 128]
        self.vst = SwinTransformer3D(**kwargs)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(768, hiddensize[0]),
            nn.BatchNorm1d(hiddensize[0]),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop),
            nn.Linear(hiddensize[0], hiddensize[1]),
            nn.BatchNorm1d(hiddensize[1]),
            nn.LeakyReLU(),
            nn.Dropout(mlp_drop),
            nn.Linear(hiddensize[1], 3),
        )

    def forward(self, x):
        x = self.mlp(self.flatten(self.vst(x)))
        return F.normalize(x)


class screening_model(nn.Module):
    def __init__(self,
                 num_classes,
                 mlp_in_channels=1024,
                 mlp_dropout_ratio=0.5,
                 patch_size=(2, 4, 4),
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 patch_norm=True,
                 ):
        super().__init__()
        self.vst = SwinTransformer3D(patch_size=patch_size,
                                     embed_dim=embed_dim,
                                     depths=depths,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_rate=drop_rate,
                                     attn_drop_rate=attn_drop_rate,
                                     drop_path_rate=drop_path_rate,
                                     patch_norm=patch_norm)
        self.pre_mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Dropout(mlp_dropout_ratio),
        )
        self.linear = nn.Linear(mlp_in_channels, num_classes)

    def forward(self, x: torch.Tensor):
        '''x: NCTHW'''
        x = self.vst(x)
        x = self.pre_mlp(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class fusion_model(nn.Module):
    def __init__(self,
                 num_classes,
                 mlp_in_channels=1024,
                 mlp_dropout_ratio=0.5,
                 freeze_vst=True,
                 patch_size=(2, 4, 4),
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 patch_norm=True,
                 sax_weight=None,
                 lax_weight=None,
                 lge_weight=None
                 ):
        super().__init__()
        weights = []
        if sax_weight is not None:
            weights.append(sax_weight)
        if lax_weight is not None:
            weights.append(lax_weight)
        if lge_weight is not None:
            weights.append(lge_weight)
        mlp_num_mod = len(weights)
        self.vst = nn.ModuleList([SwinTransformer3D(patch_size=patch_size,
                                                              embed_dim=embed_dim,
                                                              depths=depths,
                                                              num_heads=num_heads,
                                                              window_size=window_size,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              qk_scale=qk_scale,
                                                              drop_rate=drop_rate,
                                                              attn_drop_rate=attn_drop_rate,
                                                              drop_path_rate=drop_path_rate,
                                                              patch_norm=patch_norm) for _ in range(mlp_num_mod)])
        self.init_weights(weights=weights)
        self.freeze = freeze_vst
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(mlp_dropout_ratio)
        self.linear = nn.Linear(mlp_in_channels*mlp_num_mod, num_classes)
        self.freeze_vst()

    def init_weights(self, weights):
        for i in range(len(weights)):
            try:
                checkpoint = torch.load(weights[i], map_location='cpu')
                self.vst[i].load_state_dict(checkpoint, strict=False)
                print(f'{weights[i]} loaded successfully')
            except:
                print(f'{weights[i]} loading fail')

    def forward(self, x: torch.Tensor):
        '''x: NMCTHW'''
        assert x.shape[1] == len(
            self.vst), f'Multi modality input data types does not match the number of vst backbones; {len(self.vst_backbones)} types of data expected however {x.shape[1]} given'
        x = torch.permute(x, (1, 0, 2, 3, 4, 5))
        features = []
        for i in range(x.shape[0]):
            features.append(self.avgpool(
                self.vst[i](x[i])))
        features = torch.cat(features, dim=1)
        x = self.dropout(features)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    def freeze_vst(self):
        if self.freeze:
            self.vst.eval()
            for params in self.vst.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_vst()
