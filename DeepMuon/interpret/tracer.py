'''
Author: airscker
Date: 2023-02-15 20:13:01
LastEditors: airscker
LastEditTime: 2023-02-16 19:10:31
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import torch
from torch import nn
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


def neuron_tracer(model, input):
    ''''''
    nodes, _ = get_graph_node_names(model)
    fx = create_feature_extractor(model, return_nodes=nodes)
    fms = fx(input)
    info = ''
    for key in fms.keys():
        info += f'Feature: {key}\nValue:\n{fms[key]}'
