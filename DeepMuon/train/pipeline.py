import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

import numpy as np
from typing import Union
from abc import abstractmethod,ABCMeta

precision=torch.FloatTensor
class _base(metaclass=ABCMeta):
    '''
    ## Initialize the model prediction pipeline
        
    ### Args:
        - model: the model instance.
    '''
    def __init__(self,model:nn.Module) -> None:
        self.model=model
    @abstractmethod
    def predict(self,input,label,device,precision):
        '''
        ## Prediction funtion for training/testing/inferencing.
        
        ### Args:
            - input: the input of the model.
            - label: label to be used to evaluate the model's output.
            - device: where the model and input & output stored.
            - precision: the precision of input, output and model's parameters; `torch.DoubleTensor` and `torch.FloatTensor` available.
        
        ### Returns:
            - pred: the prediction result of the model.
            - label: the label used to evaluate the model's result.
        '''
        pass

class classify(_base):
    '''
    Model prediction pipeline built for normal classfication tasks such as Swin-Transformer, ResNet, Video-Swin Transformer, Vision Transformer etc.

    The predcition pipeline will give out:
        - pred: `torch.FloatTensor`/`torch.DoubleTensor`, the predicted values of classification models.
        - label: `torch.LongTensor`, the label used to evaluate the model's results.
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
    def predict(self,input,label,device,precision):
        input=input.type(precision).to(device)
        label=label.reshape(-1).to(device)
        pred=self.model(input)
        return pred,label

class regression(_base):
    '''
    Model prediction pipeline built for normal regression tasks such as ResMax.

    The predcition pipeline will give out:
        - pred: `torch.FloatTensor`/`torch.DoubleTensor`, the predicted values of classification models.
        - label: `torch.FloatTensor`/`torch.DoubleTensor`, the regression targets used to evaluate the model's results.
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
    def predict(self, input, label, device, precision):
        input=input.type(precision).to(device)
        label=label.type(precision).to(device)
        pred=self.model(input)
        return pred,label