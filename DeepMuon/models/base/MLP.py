'''
Author: airscker
Date: 2023-09-13 19:24:38
LastEditors: airscker
LastEditTime: 2023-09-16 17:23:53
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn
from .ADNBlock import ADN

class MLPBlock(nn.Module):
    '''
    ## Multilayer Perceptron Block

    ### Args:
        - dim_input: [int], input dimension.
        - dim_output: [int], output dimension.
        - hidden_sizes: [list[int]], hidden layer sizes, default: None. If nothing given then the block will be a linear layer.
        - mode: [str], building mode, default: 'NDA'. If nothing given then the block will be built with normalization, dropout and activation layers,
            ONLY combinations of alphabets `N`,`D`,`A` are supported.
        - bias: [bool], whether to use bias in linear layers, default: True.
        - normalization: [nn.Module], normalization layer, default: None. If nothing given then no normalization layer will be used.
        - norm_kwargs: [dict], kwargs for normalization layer, default: {}. If nothing given then no kwargs will be used to initialize normalization layer, only usable when normalization is not None.
        - activation: [nn.Module], activation layer, default: None. If nothing given then no activation layer will be used.
        - dropout_rate: [float], dropout rate, default: 0.0. If 0 is given then no dropout layer will be used.
        - dropout_inplace: [bool], whether to use inplace mode in dropout layer, default: False.

    ### Tips:
        - The order of layers in the block follows pattern: `linear` -> sequence specified by `mode`, here is the meaning of each letter in `mode`:
            - `N`: normalization layer.
            - `D`: dropout layer.
            - `A`: activation layer.
        - If you want to use normalization layer, you should give the class of the layer, NOT the instance of the layer, so does the activation layer.
        - The depth of the block is `len(hidden_sizes) + 1`.
    '''
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 hidden_sizes:list[int]=None,
                 mode='NDA',
                 bias:bool=True,
                 normalization:nn.Module=None,
                 norm_kwargs:dict={},
                 activation: nn.Module=None,
                 dropout_rate: float = 0.0,
                 dropout_inplace: bool = False,
                 ):
        super().__init__()
        self.normalization=normalization
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.norm_kwargs=norm_kwargs
        self.dropout_inplace=dropout_inplace
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_sizes is None:
            self.mlp=nn.Linear(dim_input,dim_output)
        else:
            self.mlp=nn.Sequential()
            hidden_sizes=[dim_input]+hidden_sizes+[dim_output]
            self.hidden_sizes=hidden_sizes
            for i in range(len(hidden_sizes)-1):
                self.mlp.add_module(f"linear{i}",nn.Linear(hidden_sizes[i],hidden_sizes[i+1],bias=bias))
                if i!=len(hidden_sizes)-2:
                    self.mlp.add_module(f"ADN{i}",ADN(dim=hidden_sizes[i+1],
                                                      mode=mode,
                                                      normalization=normalization,
                                                      norm_kwargs=norm_kwargs,
                                                      activation=activation,
                                                      dropout_rate=dropout_rate,
                                                      dropout_inplace=dropout_inplace))
    def forward(self, x:torch.Tensor):
        return self.mlp(x)
