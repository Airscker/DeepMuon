'''
Author: airscker
Date: 2023-01-31 18:03:57
LastEditors: airscker
LastEditTime: 2023-01-31 19:45:14
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


class CRYOFIRE_encoder(nn.Module):
    '''Input shape: (128,128)'''

    def __init__(self, conform_dim=8):
        super().__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.shared_mlp = nn.Sequential(
            # ---
            # Add 2/4 hidden layers here
            # ---
            nn.Linear(256, 256),
        )
        self.conform_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, conform_dim),
        )
        self.rotation_mlp = nn.Sequential(
            # ---
            # Add hidden layers here
            # ---
            nn.Linear(256+conform_dim, 6),
        )
        self.trans_mlp = nn.Sequential(
            # ---
            # Add hidden layers here
            # ---
            nn.Linear(256+conform_dim, 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn_block3(self.cnn_block2(self.cnn_block1(x)))
        x = self.flatten(x)
        y_i = self.shared_mlp(x)
        z_i = self.conform_mlp(y_i)
        concat_yz = torch.cat([y_i, z_i], 1)
        r_i = self.rotation_mlp(concat_yz)
        t_i = self.trans_mlp(concat_yz)
        return r_i, z_i, t_i


class CRYOFIRE_decoder(nn.Module):
    def __init__(self, conform_dim=8, img_size=128) -> None:
        super().__init__()
        self.pos_encode = None
        self.v_mlp = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, z_i: torch.Tensor, r_i: torch.Tensor):
        # k = self.pos_encode(k)
        embeded_input = torch.cat([z_i, r_i], 1)
        v_i = self.v_mlp(embeded_input)
        return v_i


class CRYOFIRE(nn.Module):
    def __init__(self, conform_dim=8, img_size=128) -> None:
        super().__init__()
        self.encoder = CRYOFIRE_encoder(conform_dim=conform_dim)
        self.decoder = CRYOFIRE_decoder(
            conform_dim=conform_dim, img_size=img_size)

    def forward(self, x):
        r_i, z_i, t_i = self.encoder(x)
        v_i = self.decoder(z_i, r_i)
        return v_i+t_i
