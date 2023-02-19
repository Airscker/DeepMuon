'''
Author: airscker
Date: 2023-02-13 19:20:47
LastEditors: airscker
LastEditTime: 2023-02-19 07:59:26
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance


def DataAttr(model: nn.Module, input, label_dim):
    ''''''
    ig = IntegratedGradients(model)
    attr_array = []
    delta_array = []
    for i in range(label_dim):
        attributions, delta = ig.attribute(
            input, target=i, return_convergence_delta=True)
        attr_array.append(attributions.detach().numpy())
        delta_array.append(delta.detach().numpy())
        print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)
    return np.array(attr_array), np.array(delta_array)
