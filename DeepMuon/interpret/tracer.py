'''
Author: airscker
Date: 2023-02-15 20:13:01
LastEditors: airscker
LastEditTime: 2023-02-20 00:44:43
Description: Trace the data flow within the model

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import torch
import warnings
from typing import Union
from torch import nn
from DeepMuon.tools.AirFunc import check_device

class Neuron_Flow:
    '''
    ## Trace the data flow of model's neurons

    ### Args:
        - model: the model to be traced, the tracing mechanism will be done under the evaluation mode
        - input_data: the input data of the model
        - device: the GPU to be used to inference the model

    ### Some Properties:
        - hooks: `dict(module_id = forward_hook)`
            the hooks built for every neuron to trace the data flow.
            hooks will be created once you call `trace()` method and after the inference work, hooks will be removed
        - neuron_info: `dict(module_id = [neuron_name, module_name, input_data, output_data])`
            the property records the most important info which we want, to enable this property,
            you need to call `trace()` method to record the input/output data of every neuron

    ### Tips:
        - Once you create the `Neuron_Flow` object, the tracing will be done by using data and model given, the initialization pipeline is listed as following:
            - Get model and input data
            - Run `trace()` to get information of neurons
        - If you want to test another input data, please call `trace()` again and pass your input data to it
        - You can get every neuron's input data and output data by refering property `neuron_info`
    
    ### Examples:

    >>> NeuronF=Neuron_Flow(model=screening_model(),input_data=torch.rand(1,3,13,100,100))
    >>> print(NeuronF)
    >>> Model Architecture: screening_model(
            (vst): SwinTransformer3D(
                (patch_embed): PatchEmbed3D(
                (proj): Conv3d(3, 128, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            ...
            Module ID: 2745179684480, Neuron Name: linear, Module Name: Linear
                input[0]: torch.Size([1, 1024])
                output[0]: torch.Size([11])
    >>> # Trace another flow
    >>> NeuronF.trace(torch.rand(2,3,13,120,120))
    >>> print(NeuronF)
    >>> Model Architecture: screening_model(
            (vst): SwinTransformer3D(
                (patch_embed): PatchEmbed3D(
                (proj): Conv3d(3, 128, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            ...
            Module ID: 2745179684480, Neuron Name: linear, Module Name: Linear
                input[0]: torch.Size([2, 1024])
                output[0]: torch.Size([11])
                output[1]: torch.Size([11])
    '''
    def __init__(self,model:nn.Module,input_data:torch.Tensor,device:Union[int,str,torch.device]='cpu'):
        self.device=check_device(device)
        self.model=model.to(self.device)
        self.model.eval()
        self.input=input_data.to(self.device)
        self.hooks={}
        self.neuron_info={}
        self.trace()
    def __tracer_hook(self,module,input,output):
        '''Hook built for tracing data flow'''
        self.neuron_info[id(module)]+=[input,output]
    def __clean_hooks(self):
        '''Clean hooks of every neuron''' 
        for module_id in self.hooks.keys():
            self.hooks[module_id].remove()
    def __apply_hooks(self):
        '''Apply tracing hook for every neuron recursively'''
        model_name=self.model.__class__.__name__
        stack: list[tuple[str, nn.Module, int]] = [(model_name, self.model, 0)]
        while stack:
            var_name, module, curr_depth = stack.pop()
            module_id = id(module)
            if module_id in self.hooks:
                self.hooks[module_id].remove()
            self.hooks[module_id] = module.register_forward_hook(self.__tracer_hook)
            self.neuron_info[module_id]=[var_name,module.__class__.__name__]
            for name, mod in reversed(module._modules.items()):
                if mod is not None:
                    stack += [(name, mod, curr_depth + 1)]
    def trace(self,input_data=None):
        '''
        ## Trace the data flow and record it

        ### Pipeline:
            - Register forward hooks
            - Trace the data flow and record it
            - Clean hooks registered
        '''
        if input_data is not None:
            self.input=input_data.to(self.device)
        self.__apply_hooks()
        self.model(self.input)
        self.__clean_hooks()
        try:
            torch.cuda.empty_cache()
        except:
            pass
    def __repr__(self) -> str:
        info=''
        for module_id in self.neuron_info.keys():
            info+=f"Module ID: {module_id}, Neuron Name: {self.neuron_info[module_id][0]}, Module Name: {self.neuron_info[module_id][1]}\n"
            try:
                input=self.neuron_info[module_id][2]
                output=self.neuron_info[module_id][3]
                for i in range(len(input)):
                    try:
                        info+=f"\tinput[{i}]: {input[i].shape}\n"
                    except:
                        info+=f"\tinput[{i}] is not a Tensor or NumpyArray\n"
                for i in range(len(output)):
                    try:
                        info+=f"\toutput[{i}]: {output[i].shape}\n"
                    except:
                        info+=f"\toutput[{i}] is not a Tensor or NumpyArray\n"
            except:
                pass
        return f'Model Architecture: {self.model}\nNeuron flow information:\n{info}'