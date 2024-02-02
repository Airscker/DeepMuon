'''
Author: airscker
Date: 2023-12-14 01:48:44
LastEditors: airscker
LastEditTime: 2023-12-14 01:51:25
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from dgl.nn.pytorch import GraphConv,GINConv,GINEConv
from typing import Union
from dgl.utils import expand_as_pair

from .base import MLPBlock,FourierExpansion,RadialBesselFunction,SphericalBesselFunction

class MIGNNBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        pass