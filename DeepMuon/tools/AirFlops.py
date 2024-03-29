'''
Author: airscker
Date: 2022-10-05 13:35:07
LastEditors: airscker
LastEditTime: 2023-09-03 16:08:02
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from typing import Union
from ptflops import get_model_complexity_info
from torchinfo import summary


def model_para(model, datasize: Union[list,tuple], depth=3, device='cpu', show=False):
    """
    ## Get the number of parameters and FLOPs for a given model as well as the detailed list of layers

    ### Args:
        - model: The model to be tested, such as model=MLP()
        - datasize: A list containing the size of each dimension of the input, such as [N,H,W]
        - depth: The depth at which to run summary(). Default is 3.
        - device: the id of GPU to be used, if it does not exist the CPU will be used
        - show: whether to print information of model

    ### Return:
        - flops: the FLOPS of model
        - params: the size of parameters of the model
        - sumres: the detailed report of model acrchitecture
    """
    if torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    # model = model()
    model = model.to(device)
    model.float()
    datasize=list(datasize)
    datasize[0] = 2
    flops, params = get_model_complexity_info(model, tuple(datasize[1:]), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    sumres = summary(model, input_size=tuple(
        datasize[:]), depth=depth, verbose=0, device=device)
    if show:
        print(sumres)
        print(f"Overall Model GFLOPs: {flops}, Params: {params}")
    return flops, params, sumres
