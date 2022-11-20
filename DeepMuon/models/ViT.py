'''
Author: airscker
Date: 2022-09-28 12:20:22
LastEditors: airscker
LastEditTime: 2022-10-09 14:10:45
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from typing import Sequence, Union

torch.set_default_tensor_type(torch.DoubleTensor)

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class Vit_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=ViT(3,[10,10,40],[10,10,20],hidden_size=256,num_layers=3,num_heads=16,mlp_dim=1024)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(2*256, 1024),
            nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            HailingDirectNorm()
        )
    def forward(self,x):
        x=self.vit(x)
        x=self.flatten(x)
        x=self.mlp(x)
        return x

class Vit_MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=ViT(3,[10,10,40],[10,10,20],hidden_size=128,num_layers=3,num_heads=16,mlp_dim=32)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(2*128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            HailingDirectNorm()
        )
    def forward(self,x):
        x=self.vit(x)
        x=self.flatten(x)
        x=self.mlp(x)
        return x

class Vit_MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=ViT(3,[10,10,40],[5,5,10],hidden_size=16,num_layers=12,num_heads=4,mlp_dim=16)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(16*16, 64),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 3),
            HailingDirectNorm()
        )
    def forward(self,x):
        x=self.vit(x)
        x=self.flatten(x)
        x=self.mlp(x)
        return x

class HailingDirectNorm(nn.Module):
    def __init__(self) -> None:
        '''
        ## Customized Layer, Normalize the Direction Vector of Hailing Data Derived from _Direct Models
        - Input: [N,3], info: [px,py,pz]
        - Output: [N,3], info: [px,py,pz](Normalized)

        N is the batch size, and the output direction vector is normalized to 1
        '''
        super().__init__()
    def forward(self,x):
        return F.normalize(x)