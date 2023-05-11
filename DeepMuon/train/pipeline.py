import torch
from torch import nn
from abc import abstractmethod,ABCMeta

class Pipeline(metaclass=ABCMeta):
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
        pred=self.model(input)
        return pred,label

class classify(Pipeline):
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

class regression(Pipeline):
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
    
class cnnlstm_cla(Pipeline):
    '''
    Model prediction pipeline built for CNNLSTM which was implemented within DeepMuon
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
    def predict(self, input, label, device, precision):
        label = label.reshape(-1)
        # if num_frames is not None:
        #     label = np.repeat(label, num_frames)
        if isinstance(input, list):
            input = [torch.autograd.Variable(x_).type(precision).cuda(
                device, non_blocking=True) for x_ in input]
            h0 = self.model.module.init_hidden(input[0].size(0))
        else:
            input = torch.autograd.Variable(input).type(precision).cuda(device, non_blocking=True)
            h0 = self.model.module.init_hidden(input.size(0))
        label = torch.autograd.Variable(label).cuda(device, non_blocking=True)
        pred=self.model(input,h0)
        return pred,label