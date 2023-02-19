'''
Author: airscker
Date: 2023-02-15 20:13:01
LastEditors: airscker
LastEditTime: 2023-02-19 13:41:50
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import torch
from torch import nn
from typing import Any
from DeepMuon.models import *

class Neuron_Flow:
    '''
    ## Trace the data flow of model's neurons

    ### Args:
        - model: the model to be traced
        - input_data: the input data of the model
        - mode: `eval`/`train`, whether to cancel the gradient refreshing and freezing normalization layer(eg. BatchNorm)

    ### Properties:
        - hooks: dict(module_id = forward_hook), the hooks built for every neuron to trace the data flow.
            hooks will be created once you call `trace()` method and after the inference work, hooks will be removed
        - neuron_info: dict(module_id = [neuron_name, module_name, input_data, output_data]), to enable this property,
            you need to call `trace()` method to record the input/output data of every neuron
    '''
    def __init__(self,model:nn.Module,input_data,mode='eval'):
        self.model=model
        if mode=='eval':
            self.model.eval()
        elif mode=='train':
            self.model.train()
        self.input=input_data
        self.hooks={}
        self.neuron_info={}
        self.trace()
    def __tracer_hook(self,module,input,output):
        self.neuron_info[id(module)]+=[input,output]
    def __clean_hooks(self):
        '''Clean hooks of every neuron''' 
        for module_id in self.hooks.keys():
            self.hooks[module_id].remove()
    def __apply_hooks(self):
        model_name=self.model.__class__.__name__
        stack: list[tuple[str, nn.Module, int, None]] = [(model_name, self.model, 0)]
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
            self.input=input_data
        self.__apply_hooks()
        self.model(self.input)
        self.__clean_hooks()
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