'''
Author: airscker
Date: 2023-02-13 19:20:47
LastEditors: airscker
LastEditTime: 2023-02-27 11:37:57
Description: Get data/neuron/layer attributions

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
from torch import nn
import numpy as np
from DeepMuon.tools.AirFunc import check_device
from typing import Union, Tuple, Callable
from captum.attr import IntegratedGradients, NeuronConductance, LayerConductance, GuidedGradCam, GradientShap


def GradCAM(model: nn.Module, module: nn.Module, input: torch.Tensor, label_dim: int, device: Union[int, str, torch.device] = 'cpu'):
    '''
    ## Get data attribution using GuidedGradCAM method
        To get more details about GuidedGradCAM algorithm, please refer to https://arxiv.org/abs/1610.02391

    ### Args:
        - model: The model to be interpreted
        - module: For which GradCAM attributions are computed.
        - input: The input data of model, make sure its `requires_grad` property is set as `True`
        - label_dim: The dimension of model's output, eg. binary classification task's dimension is 2
        - device: the GPU to be used to inference the model

    ### Return: np.ndarray
        - attr_array: np.ndarray, contains all attribution for every target
            Element-wise product of (upsampled) GradCAM and Guided Backprop attributions.
            Attributions will be the same size as the provided inputs, with each value providing the attribution of the corresponding input index.
            If the GradCAM attributions cannot be upsampled to the shape of a given input tensor, None is returned in the corresponding index position.

    ### Tips:
        - module: The module given here should be one of the layers of the model
            eg. We can specify `module=model.conv1`
    '''
    device = check_device(device)
    model.eval()
    model.to(device)
    module.to(device)
    input.requires_grad = True
    input = input.to(device)
    guided_gc = GuidedGradCam(model, module)
    attr_array = []
    for i in range(label_dim):
        attr_array.append(guided_gc.attribute(input, i).detach().cpu().numpy())
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return np.array(attr_array)


def ItegGrad(model: nn.Module, input: torch.Tensor, label_dim: int, device: Union[int, str, torch.device] = 'cpu'):
    '''
    ## Get data attribution using Integrated Gradient method
        To get more details about IntegratedGradient algorithm, please refer to https://arxiv.org/abs/1703.01365

    ### Args:
        - model: The model to be interpreted
        - input: The input data of model, make sure its `requires_grad` property is set as `True`
        - label_dim: The dimension of model's output, eg. binary classification task's dimension is 2
        - device: the GPU to be used to inference the model

    ### Return: tuple(np.ndarray, np.ndarray, list)
        - attr_array: np.ndarray, contains all attribution for every target
            Attributions will always be the same size as the provided inputs, with each value providing the attribution of the corresponding input index.
        - delta_array: np.nparray, contains all convergence delta values for every target
        - convergence: list[bool], contains all deltas' convergency for every target
    '''
    device = check_device(device)
    model.eval()
    model.to(device)
    input.requires_grad = True
    input = input.to(device)
    ig = IntegratedGradients(model)
    attr_array = []
    delta_array = []
    convergence = []
    for i in range(label_dim):
        attributions, delta = ig.attribute(
            input, target=i, return_convergence_delta=True)
        attr_array.append(attributions.detach().cpu().numpy())
        delta_array.append(delta.detach().cpu().numpy())
        convergence.append(ig.has_convergence_delta())
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return np.array(attr_array), np.array(delta_array), convergence


def NeuronCond(model: nn.Module, module: nn.Module, input: torch.Tensor, neuron_index: Union[int, Tuple[int, ...], Callable], device: Union[int, str, torch.device] = 'cpu'):
    '''
    ## Get neuron conductance using NeuronConductance method
        To get more details about NeuronConductance algorithm, please refer to https://arxiv.org/abs/1805.12233

    ### Args:
        - model: The model to be interpreted
        - module: The model's layer element to be interpreted
        - input: The input data of model
        - neuron_index: the index of neuron within the specified module
        - device: the GPU to be used to inference the model

    ### Return: np.ndarray
        - attr_array: np.ndarray, conductance of the selected neuron
             Attributions will always be the same size as the provided inputs, with each value providing the attribution of the corresponding input index.

    ### Tips:
        - module: The module given here should be one of the layers of the model, and the neuron we are interested in is contained within this layer.
            eg. We can specify `module=model.conv1`
        - neuron_index: To compute neuron attribution, we need to provide the neuron index for which attribution is desired.
            Suppose the layer output is N x 12 x 32 x 32, we need a tuple in the form (0..11, 0..31, 0..31) which indexes a particular neuron in the layer output.
            For this example, we choose the index (4,1,2) to computes neuron gradient for neuron with index (4,1,2).
    '''
    device = check_device(device)
    model.eval()
    module.eval()
    model.to(device)
    module.to(device)
    input.requires_grad = True
    input = input.to(device)
    neuron_cond = NeuronConductance(model, module)
    neuron_attr = neuron_cond.attribute(
        inputs=input, neuron_selector=neuron_index)
    res = neuron_attr.detach().cpu().numpy()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return res


def LayerCond(model: nn.Module, module: nn.Module, input: torch.Tensor, label_dim: int, device: Union[int, str, torch.device] = 'cpu'):
    '''
    ## Get layer attribution using LayerConductance method
        To get more details about LayerConductance algorithm, please refer to https://arxiv.org/abs/1805.12233, https://arxiv.org/abs/1807.09946

    ### Args:
        - model: The model to be interpreted
        - module: The model's layer element to be interpreted
        - input: The input data of model, make sure its `requires_grad` property is set as `True`
        - label_dim: The dimension of model's output, eg. binary classification task's dimension is 2
        - device: the GPU to be used to inference the model

    ### Return: tuple(np.ndarray, np.ndarray, list)
        - attr_array: np.ndarray, contains all attribution for every target
            Attributions will always be the same size as the input or output of the given layer,
            depending on whether we attribute to the inputs or outputs of the layer which is decided by the input flag `attribute_to_layer_input`.
            Here `attribute_to_layer_input` is set to True then the attributions will be computed with respect to layer inputs, 
            otherwise it will be computed with respect to layer outputs.
        - delta_array: np.nparray, contains all convergence delta values for every target
        - convergence: list[bool], contains all deltas' convergency for every target

    ### Tips:
        - module: The module given here should be one of the layers of the model, and the neuron we are interested in is contained within this layer.
            eg. We can specify `module=model.conv1`
    '''

    device = check_device(device)
    model.eval()
    module.eval()
    model.to(device)
    module.to(device)
    input.requires_grad = True
    input = input.to(device)
    layer_cond = LayerConductance(model, module)
    attr_array = []
    delta_array = []
    convergence = []
    for i in range(label_dim):
        attributions, delta = layer_cond.attribute(
            input, target=i, return_convergence_delta=True)
        attr_array.append(attributions.detach().cpu().numpy())
        delta_array.append(delta.detach().cpu().numpy())
        convergence.append(layer_cond.has_convergence_delta())
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return np.array(attr_array), np.array(delta_array), convergence

def GradShap(model: nn.Module,input: torch.Tensor, label_dim: int,baselines: torch.Tensor=None,multiply_inputs:bool=True, device: Union[int, str, torch.device] = 'cpu'):
    '''
    ## Get layer attribution using GradientShap method
        To get more details about GradientShap algorithm, please refer to https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

    ### Args:
        - model: The model to be interpreted
        - input: The input data of model, make sure its `requires_grad` property is set as `True`
        - label_dim: The dimension of model's output, eg. binary classification task's dimension is 2
        - baseline: Baselines define the starting point from which expectation is computed
        - multiply_by_inputs:
            - Indicates whether to factor model inputs' multiplier in the final attribution scores. In the literature this is also known as local vs global attribution.
            - If inputs' multiplier isn't factored in then this type of attribution method is also called local attribution. If it is, then that type of attribution method is called global.
            - More detailed can be found here: https://arxiv.org/abs/1711.06104
            - In case of gradient shap, if multiply_by_inputs is set to True, the sensitivity scores of scaled inputs are being multiplied by (inputs - baselines).
        - device: the GPU to be used to inference the model

    ### Return: tuple(np.ndarray, np.ndarray, list)
        - attr_array: np.ndarray, contains all attribution for every target
            Attributions will always be the same size as the input or output of the given layer,
            depending on whether we attribute to the inputs or outputs of the layer which is decided by the input flag `attribute_to_layer_input`.
            Here `attribute_to_layer_input` is set to True then the attributions will be computed with respect to layer inputs, 
            otherwise it will be computed with respect to layer outputs.
        - delta_array: np.nparray, contains all convergence delta values for every target
        - convergence: list[bool], contains all deltas' convergency for every target
    '''
    device=check_device(device)
    model.eval()
    model.to(device)
    input.requires_grad=True
    input=input.to(device)
    gradshap=GradientShap(model,multiply_by_inputs=multiply_inputs)
    attr_array = []
    delta_array = []
    convergence = []
    if baselines is None:
        base_shape=list(input.shape)
        base_shape[0]=20
        baselines=torch.randn(base_shape).to(device)
    for i in range(label_dim):
        attr,delta=gradshap.attribute(inputs=input,baselines=baselines,target=i,return_convergence_delta=True)
        attr_array.append(attr.detach().cpu().numpy())
        delta_array.append(delta.detach().cpu().numpy())
        convergence.append(gradshap.has_convergence_delta())
    return np.array(attr_array),np.array(delta_array),convergence
