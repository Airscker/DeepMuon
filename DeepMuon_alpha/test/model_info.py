'''
Author: airscker
Date: 2022-10-05 13:35:07
LastEditors: airscker
LastEditTime: 2022-10-05 13:56:17
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
from torch import nn
import torch
from ptflops import get_model_complexity_info
from torchinfo import summary
torch.set_default_tensor_type(torch.FloatTensor)

def model_para(model,datasize:list,depth=3,gpu=0):
    """Get model information

    Args:
        model ([type]): Must be a model pointer
        datasize (list): The size of the data to be fed into the model:[batch_size,other_size]
        depth (int, optional): The depth to analysis the model. Defaults to 3.
        gpu (int, optional): The id of the GPU. Defaults to 0.

    Returns:
        [type]: [description]
    """
    device=torch.device(gpu)
    model=model()
    model=model.to(device)
    model.float()
    datasize[0]=2
    flops, params = get_model_complexity_info(model, tuple(datasize[1:]), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    sumres=summary(model,input_size=tuple(datasize[:]),depth=depth,verbose=1,device=device)
    return flops,params,sumres