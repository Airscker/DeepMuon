# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
'''
Author: airscker
Date: 2023-06-08 10:51:07
LastEditors: airscker
LastEditTime: 2023-12-11 15:16:40
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn
from ..base import SwinTransformer

class EGFR_SwinT(nn.Module):

    def __init__(self,
                 cla_drop=0.2,
                 num_classes=2,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=[3],
                 frozen_stages=-1,
                 use_checkpoint=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swinT = SwinTransformer(pretrain_img_size=pretrain_img_size,
                                     patch_size=patch_size,
                                     in_chans=in_chans,
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
                                     norm_layer=norm_layer,
                                     ape=ape,
                                     patch_norm=patch_norm,
                                     out_indices=out_indices,
                                     frozen_stages=frozen_stages,
                                     use_checkpoint=use_checkpoint)
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.classify = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Dropout(cla_drop), nn.Flatten(),
            nn.Linear(self.num_features, num_classes))

    def forward(self, x: torch.Tensor):
        '''x: NCHW'''
        swint_feature = self.swinT(x)[0]
        '''Classify'''
        cla_feature = self.classify(swint_feature)
        return cla_feature


class EGFR_SwinTV2(nn.Module):

    def __init__(self,
                 cla_drop=0.3,
                 num_classes=2,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=[3],
                 frozen_stages=-1,
                 use_checkpoint=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swinT = SwinTransformer(pretrain_img_size=pretrain_img_size,
                                     patch_size=patch_size,
                                     in_chans=in_chans,
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
                                     norm_layer=norm_layer,
                                     ape=ape,
                                     patch_norm=patch_norm,
                                     out_indices=out_indices,
                                     frozen_stages=frozen_stages,
                                     use_checkpoint=use_checkpoint)
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.classify = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Dropout(cla_drop), nn.Flatten(),
            nn.Linear(self.num_features, num_classes))
        self.recon = nn.Sequential(
            nn.ConvTranspose2d(self.num_features,
                               int(self.num_features / 2),
                               2,
                               stride=2),
            nn.BatchNorm2d(int(self.num_features / 2)), nn.LeakyReLU(),
            nn.ConvTranspose2d(int(self.num_features / 2),
                               int(self.num_features / 8),
                               2,
                               stride=2),
            nn.BatchNorm2d(int(self.num_features / 8)), nn.LeakyReLU(),
            nn.ConvTranspose2d(int(self.num_features / 8),
                               int(self.num_features / 32),
                               2,
                               stride=2),
            nn.BatchNorm2d(int(self.num_features / 32)), nn.LeakyReLU(),
            nn.ConvTranspose2d(int(self.num_features / 32),
                               int(self.num_features / 64),
                               2,
                               stride=2),
            nn.BatchNorm2d(int(self.num_features / 64)), nn.LeakyReLU(),
            nn.ConvTranspose2d(int(self.num_features / 64),
                               int(self.num_features / 256),
                               2,
                               stride=2),
            nn.BatchNorm2d(int(self.num_features / 256)), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        '''x: NCHW'''
        swint_feature = self.swinT(x)[0]
        '''Classify'''
        cla_feature = self.classify(swint_feature)
        '''Reconstruction'''
        rec_feature = self.recon(swint_feature)
        '''Feature Alignment'''
        ali_feature = self.swinT(rec_feature)[0]
        return cla_feature, rec_feature, ali_feature, swint_feature
