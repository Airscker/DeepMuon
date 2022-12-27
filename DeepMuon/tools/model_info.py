'''
Author: airscker
Date: 2022-10-05 13:35:07
LastEditors: airscker
LastEditTime: 2022-12-27 17:12:52
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import torch
from ptflops import get_model_complexity_info
from torchinfo import summary
torch.set_default_tensor_type(torch.FloatTensor)


def model_para(model, datasize: list, depth=3, gpu=0, show=False):
    """
    The model_para function returns the number of parameters and FLOPs for a given model.
    It takes in the following arguments:
    model - The model to be tested. Must have no more than 3 layers, as otherwise it will not work with this function. 
    datasize - A list containing the size of each dimension of an input image (in order height, width, channels). 
    depth - The depth at which to run summary(). Default is 3. If you are using a custom class that has different naming conventions from ResNet-like classes, then you may need to increase depth by 1 or 2 in order for all layer

    :param model: Define the model
    :param datasize:list: Specify the input size of the model
    :param depth=3: Determine the number of layers in the model
    :param gpu=0: Specify which gpu to use
    :return: The flops, params and summary of the model
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
