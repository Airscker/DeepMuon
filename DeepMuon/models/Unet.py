'''
Author: airscker
Date: 2022-12-27 16:37:52
LastEditors: airscker
LastEditTime: 2023-02-09 09:57:40
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
from monai.networks.nets import UNet
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit
torch.set_default_tensor_type(torch.DoubleTensor)


class ResMax_base(nn.Module):
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [4, 3, 2, 1]
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

    def forward(self, x: torch.Tensor):
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


class UNet_VAE(nn.Module):
    def __init__(self, mlp_drop_rate=0, resmax_path: str = None) -> None:
        super().__init__()
        self.unet = UNet(spatial_dims=3, in_channels=3,
                         out_channels=3, channels=[2, 4, 8, 16], strides=[1, 1, 1], num_res_units=3)
        self.resmax = ResMax_base(mlp_drop_rate=mlp_drop_rate)
        if resmax_path is not None:
            self.init_resmax(resmax_path)

    def init_resmax(self, checkpoint: str):
        model_dict = torch.load(checkpoint)['model']
        self.resmax.load_state_dict(model_dict)
        print(f'ResMax initialized from {checkpoint}')

    def freeze_resmax(self):
        self.resmax.eval()
        for params in self.resmax.parameters():
            params.requires_grad = False

    def freeze_unet(self):
        self.unet.eval()
        for params in self.unet.parameters():
            params.requires_grad = False

    def train(self, mode: bool = True):
        # self.freeze_resmax()
        self.freeze_unet()
        super().train(mode)

    def forward(self, x: torch.Tensor):
        x = self.unet(x)
        x = self.resmax(x)
        return x


class UNet_VAE2(nn.Module):
    def __init__(self, mlp_drop_rate=0) -> None:
        super().__init__()
        self.unet = UNet(spatial_dims=3, in_channels=3,
                         out_channels=3, channels=[2, 4, 8, 16], strides=[1, 1, 1], num_res_units=3)
        self.pool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.hidden_size = [512, 256]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*6*6*6, self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 3),
        )

    def freeze_mlp(self):
        self.mlp.eval()
        for params in self.mlp.parameters():
            params.requires_grad = False

    def freeze_unet(self):
        self.unet.eval()
        for params in self.unet.parameters():
            params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # self.freeze_resmax()
        # self.freeze_unet()

    def forward(self, x: torch.Tensor):
        x = self.unet(x)
        x = self.mlp(self.pool(x))
        return x
