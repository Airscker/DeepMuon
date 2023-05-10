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

def train(device: Union[int, str, torch.device],
          dataloader: DataLoader,
          model: nn.Module,
          loss_fn=None,
          optimizer=None,
          scheduler=None,
          gradient_accumulation: int = 8,
          grad_clip: float = None,
          fp16: bool = False,
          grad_scalar: GradScaler = None):
    '''
    ## Train model and refrensh its gradients & parameters

    ### Tips:
        - Gradient accumulation: Gradient accumulation steps
        - Mixed precision: Mixed precision training is allowed
        - Gradient resacle: Only available when mixed precision training is enabled, to avoid the gradient exploration/annihilation bring by fp16
        - Gradient clip: Using gradient value clip technique
    '''
    model.train()
    train_loss = 0
    predictions = []
    labels = []
    batchs = len(dataloader)
    gradient_accumulation = min(batchs, gradient_accumulation)
    for i, (x, y) in enumerate(dataloader):
        x, y = x.type(precision).to(device), y.to(device)
        with autocast(enabled=fp16):
            if (i+1) % gradient_accumulation != 0:
                with model.no_sync():
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    loss = loss/gradient_accumulation
                    grad_scalar.scale(loss).backward()
            elif (i+1) % gradient_accumulation == 0:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss = loss/gradient_accumulation
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), grad_clip)
                grad_scalar.scale(loss).backward()
                grad_scalar.step(optimizer)
                grad_scalar.update()
                optimizer.zero_grad()
        predictions.append(pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        train_loss += loss.item()*gradient_accumulation
    scheduler.step()
    return train_loss/batchs, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


def test(device, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.type(precision).to(device), y.to(device)
            pred = model(x)
            predictions.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)
