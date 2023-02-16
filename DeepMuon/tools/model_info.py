'''
Author: airscker
Date: 2022-10-05 13:35:07
LastEditors: airscker
LastEditTime: 2023-02-16 18:09:48
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from ptflops import get_model_complexity_info
from torchinfo import summary
torch.set_default_tensor_type(torch.FloatTensor)


def model_para(model, datasize: list, depth=3, gpu=0, show=False):
    """
    ## Get the number of parameters and FLOPs for a given model as well as the detailed list of layers

    ### Args:
        - model: The model to be tested.
        - datasize: A list containing the size of each dimension of the input
        - depth: The depth at which to run summary(). Default is 3.
        - gpu: the id of GPU to be used, if it does not exist the CPU will be used
        - show: whether to print information of model

    ### Return:
        - flops: the FLOPS of model
        - params: the size of parameters of the model
        - sumres: the detailed report of model acrchitecture
    """
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')
    # model = model()
    model = model.to(device)
    model.float()
    datasize[0] = 2
    flops, params = get_model_complexity_info(model, tuple(datasize[1:]), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    sumres = summary(model, input_size=tuple(
        datasize[:]), depth=depth, verbose=0, device=device)
    if show:
        print(sumres)
        print(f"Overall Model GFLOPs: {flops}, Params: {params}")
    return flops, params, sumres
