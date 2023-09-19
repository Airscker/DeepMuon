'''
Author: airscker
Date: 2023-09-16 17:11:34
LastEditors: airscker
LastEditTime: 2023-09-16 19:23:29
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn

class ADN(nn.Module):
    '''
    ## Activation-Dropout-Normalization sequence adjustable block.

    ### Args:
        - dim: [int], the dimension of the input data.
        - mode: [str], the sequence of the block, only 'N' 'D' 'A' are supported.
        - normalization: [nn.Module], the normalization layer. If `None` is given, no normalization layer will be added.
        - norm_kwargs: [dict], the kwargs of the normalization layer.
        - activation: [nn.Module], the activation layer. If `None` is given, no activation layer will be added.
        - dropout_rate: [float], the dropout rate. If `0` is given, no dropout layer will be added.
        - dropout_inplace: [bool], whether to set the dropout layer to inplace mode.
    '''
    def __init__(self,
                 dim:int=0,
                 mode='NDA',
                 normalization:nn.Module=None,
                 norm_kwargs:dict={},
                 activation: nn.Module=None,
                 dropout_rate: float = 0.0,
                 dropout_inplace: bool = False,
                 ) -> None:
        super().__init__()
        building_mode={'N':self.__build_norm,'A':self.__build_act,'D':self.__build_dropout}
        self.ADN=nn.Sequential()
        self.dim=dim
        self.normalization=normalization
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.norm_kwargs=norm_kwargs
        self.dropout_inplace=dropout_inplace
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        for layer in mode:
            assert layer in building_mode.keys(),f"Unknown building mode {layer} given, only 'N' 'D' 'A' are supported. Please check it and try again."
            building_mode[layer]()
    def __build_norm(self):
        if self.normalization is not None:
            self.ADN.add_module(f"norm",self.normalization(self.dim,**self.norm_kwargs))
    def __build_act(self):
        if self.activation is not None:
            self.ADN.add_module(f"act",self.activation())
    def __build_dropout(self):
        if self.dropout_rate>0:
            self.ADN.add_module(f"dropout",nn.Dropout(self.dropout_rate,inplace=self.dropout_inplace))
    def forward(self, data:torch.Tensor) -> torch.Tensor:
        return self.ADN(data)