'''
Author: Airscker
Date: 2022-08-25 22:02:01
LastEditors: airscker
LastEditTime: 2022-09-21 09:32:55
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import time
import os
from typing import Any, List
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import click
import numpy as np

import AirFunc
from models import *
from dataset import *
import AirLogger

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution,NeuronConductance,LayerConductance,DeepLift
from captum.attr import visualization as viz

from nni.experiment import Experiment

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models
from ptflops import get_model_complexity_info
from torchsummary import summary
torch.set_default_tensor_type(torch.FloatTensor)


def MLP3Test(test_data='../Pandax-4T-PosRec/data/IMG2D_XY_test.pkl',batch_size=10000,modelpath='../Pandax-4T-PosRec/models/MLP3_3_best/Best_Performance.pth'):
    # shape of the input data is [N,1,17,17]
    device=torch.device('cuda:0')
    test_dataset=PandaxDataset(IMG_XY_path=test_data)
    test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
    model=MLP3()
    out=AirFunc.load_model(modelpath,device=device)
    model.load_state_dict(out[1],False)
    model.to(device=device)
    # Integ_Grad(model=model,device=device)
    # loss_fn=nn.MSELoss()
    # loss_fn=MSALoss()
    # print(loss_fn.state_dict())
    # print('Testing data...')
    # ave_loss,loss=single_test(device=device,dataset=test_dataset,model=model,loss_fn=loss_fn)
    # ave_loss=test(device=device,dataloader=test_dataloader,model=model,loss_fn=loss_fn)
    # print(f'Average loss value of test dataset: {ave_loss}')
    # AirFunc.hist_plot(loss,inter=1e-3)
    # np.save('res.npy',loss)


    return 0

def model_para(model:nn.Module,datasize:List):
    device=torch.device('cuda:0')
    model=model.to(device)
    data=torch.randn(datasize)
    data=data.to(device)
    out=model(data)
    # print(data,'\n',out)
    print(f'Output size of the model: {out.shape}(First dimension is batch_size)')
    flops, params = get_model_complexity_info(model, tuple(datasize[1:]), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print(summary(model,input_size=tuple(datasize[1:]),batch_size=1))
    # flops,params=profile(model=model,inputs=data)
    print(f"Overall Model GFLOPs: {flops}, Params: {params}")
    # return flops,params


def Integ_Grad(model:nn.Module,device:torch.device,index=0):
    '''
    The algorithm outputs an attribution score for each input element and a convergence delta. \
        The lower the absolute value of the convergence delta the better is the approximation.

    Positive attribution score means that the input in that particular position positively contributed to the final prediction and negative means the opposite. \
        The magnitude of the attribution score signifies the strength of the contribution. \
            Zero attribution score means no contribution from that particular feature.
    '''
    model.to(device)
    model.eval()
    print(model)
    dataset=AirFunc.PandaxTensorData(IMG_XY_path='../Pandax-4T-PosRec/data/IMG2D_XY.pkl')
    ig=IntegratedGradients(model)
    # ig=DeepLift(model)
    data=dataset.gettensor(index)
    attr,delta=ig.attribute(data[0].to(device),target=0,return_convergence_delta=True)
    print('Integrated Gradient Distribution for X')
    print('IG Attributions:', attr)
    print('Convergence Delta:', delta)
    attr,delta=ig.attribute(data[0].to(device),target=1,return_convergence_delta=True)
    print('Integrated Gradient Distribution for Y')
    print('IG Attributions:', attr)
    print('Convergence Delta:', delta)
    
    plt.figure(figsize=(15,8))
    plt.subplot(121)
    plt.title('Original data')
    plt.imshow(dataset.getitem(index)[0],cmap='jet')
    plt.colorbar()
    plt.subplot(122)
    plt.title('Contribution Heat Map')
    plt.imshow(attr.detach().cpu().numpy()[0],cmap='jet')
    plt.colorbar()
    plt.show()
    # layers=model.modules()
    # for i in range(len(layers)):
    #     nc = NeuronConductance(model, model.lin1)
    #     attributions = nc.attribute(input, neuron_selector=1, target=0)
    #     print('Neuron Attributions:', attributions)
    #     lc = LayerConductance(model, model.lin1)
    #     attributions, delta = lc.attribute(input, target=0, return_convergence_delta=True)
    #     print('Layer Attributions:', attributions)
    #     print('Convergence Delta:', delta)




def test(device,dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss

def single_test(device,dataset,model,loss_fn):
    """Test data one by one and get their predicted value and loss

    Args:
        device (torch.device)\n
        dataset (torch.dataset)\n
        model\n
        loss_fn\n
    Returns:
        average loss on entire dataset\n
        the array of every sample's loss value
    """
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=True,pin_memory=True)
    model.eval()
    loss=[]
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred,y)
            # print(loss_fn(pred, y).item())
            loss.append(loss_fn(pred, y).item())
    loss=np.array(loss)
    return np.mean(loss),loss

def model_optim():
    search_space = {
    'f1': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'f2': {'_type': 'choice', '_value': [128,256,512,1024]},
    'lr':{'_type':'loguniform','_value':[0.0003,0.00001]},
    }
    print(f'Current PID: {os.getpid()}')
    experiment=Experiment('local')
    experiment.config.trial_command='python train.py'
    experiment.config.trial_code_directory='.'
    experiment.config.search_space=search_space
    experiment.config.tuner.name='TPE'
    experiment.config.tuner.class_args['optimize_mode']='minimizes'
    experiment.config.max_trial_number=20
    experiment.config.trial_concurrency=2
    experiment.run(8008)
    experiment.stop()
    return 0
start=time.time()
# MLP3Test()
model_para(MLP3_3D_Direc(),datasize=[3,10,10,40,3])
# model_para(MLP3(),datasize=[3,1,17,17])
# model_optim()
# model=MLP3()
# print(model._get_name())
print(f'Total Time used: {time.time()-start}s')
# model_para(model=MLP3v2(),datasize=[3,1,17,17])
# logger=AirLogger.LOGT()
# logger.log(f'yes{1e4}')
# logger.log(f'{time.ctime()}')


