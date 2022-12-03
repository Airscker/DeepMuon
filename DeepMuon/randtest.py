'''
Author: Airscker
Date: 2022-08-25 22:02:01
LastEditors: airscker
LastEditTime: 2022-12-03 22:53:14
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import shutil
import time
import os
from typing import Any, List
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import click
import numpy as np

from DeepMuon.tools import *
import DeepMuon.tools.AirFunc as AirFunc
from DeepMuon.models import *
from DeepMuon.dataset import *
import DeepMuon.tools.AirLogger as AirLogger

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, NeuronConductance, LayerConductance, DeepLift
from captum.attr import visualization as viz
from monai.networks.blocks import *
from monai.networks.nets import *

# from nni.experiment import Experiment

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models
from ptflops import get_model_complexity_info
from torchinfo import summary



torch.set_default_tensor_type(torch.FloatTensor)


def MLP3Test(test_data='../Pandax-4T-PosRec/data/IMG2D_XY_test.pkl', batch_size=10000, modelpath='../Pandax-4T-PosRec/models/MLP3_3_best/Best_Performance.pth'):
    # shape of the input data is [N,1,17,17]
    device = torch.device('cuda:0')
    test_dataset = PandaxDataset(IMG_XY_path=test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    model = MLP3()
    out = AirFunc.load_model(modelpath, device=device)
    model.load_state_dict(out[1], False)
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


def model_para(model: nn.Module, datasize: List, depth=3):
    device = torch.device('cuda:0')
    model = model.to(device)
    print(f'Model Architecture: {model}')
    data = torch.randn(datasize)
    data = data.to(device)
    out = model(data)
    # print(f'Input Data: {data}\nOutput Data: {out}')
    print(
        f'Output size of the model: {out[0].shape}(First dimension is batch_size)')
    flops, params = get_model_complexity_info(model, tuple(datasize[1:]), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    sumres = summary(model, input_size=tuple(
        datasize[:]), depth=depth, verbose=1)
    print(f"Overall Model GFLOPs: {flops}, Params: {params}")
    return flops, params


def Integ_Grad(model: nn.Module, device: torch.device, index=0):
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
    dataset = AirFunc.PandaxTensorData(
        IMG_XY_path='../Pandax-4T-PosRec/data/IMG2D_XY.pkl')
    ig = IntegratedGradients(model)
    # ig=DeepLift(model)
    data = dataset.gettensor(index)
    attr, delta = ig.attribute(data[0].to(
        device), target=0, return_convergence_delta=True)
    print('Integrated Gradient Distribution for X')
    print('IG Attributions:', attr)
    print('Convergence Delta:', delta)
    attr, delta = ig.attribute(data[0].to(
        device), target=1, return_convergence_delta=True)
    print('Integrated Gradient Distribution for Y')
    print('IG Attributions:', attr)
    print('Convergence Delta:', delta)

    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.title('Original data')
    plt.imshow(dataset.getitem(index)[0], cmap='jet')
    plt.colorbar()
    plt.subplot(122)
    plt.title('Contribution Heat Map')
    plt.imshow(attr.detach().cpu().numpy()[0], cmap='jet')
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


def test(device, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss


def single_test(device, dataset, model, loss_fn):
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
    test_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, pin_memory=True)
    model.eval()
    loss = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred,y)
            # print(loss_fn(pred, y).item())
            loss.append(loss_fn(pred, y).item())
    loss = np.array(loss)
    return np.mean(loss), loss


def model_optim():
    search_space = {
        'f1': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
        'f2': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
        'lr': {'_type': 'loguniform', '_value': [0.0003, 0.00001]},
    }
    print(f'Current PID: {os.getpid()}')
    # experiment = Experiment('local')
    experiment = None
    experiment.config.trial_command = 'python train.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 20
    experiment.config.trial_concurrency = 1
    experiment.run(8008)
    experiment.stop()
    return 0


# start=time.time()
# MLP3Test()
# model_para(UNETR(in_channels=3, out_channels=1,img_size=(16,16,16)),datasize=[1,3,16,16,16])
# model_para(UNETR(in_channels=3, out_channels=1,img_size=(10,10,40)),datasize=[1,3,10,10,40])
# model_para(CSPP(),datasize=[2,3,10,10,40])
# model_para(SABlock(hidden_size=30,num_heads=3),datasize=[1,1,30])
# model_para(ViT(3,[10,10,40],[10,10,40],hidden_size=3,num_layers=12,num_heads=3,mlp_dim=1024),datasize=[1,3,10,10,40])
# model_para(UNET_MLP(),datasize=[2,3,10,10,40],depth=5)
# model_para(MLPBlock(3,32,act='LeakyRELU'))
# model_para(SPP(),datasize=[1,3,10,10,40])
# model_para(LinearRegression3D(),datasize=(2,3,10,10,40))
# model_para(UNET_3D(),datasize=(2,3,10,10,40))
# model_para(SwinUNETR(img_size=[128,128,32],in_channels=3,out_channels=1,depths=(2,2,2,2),num_heads=(3,6,12,24),feature_size=24,spatial_dims=3),datasize=[1,3,128,128,32])
# spmodel_para(spconv.SparseConv3d(3,3,3),datasize=(1,3,10,10,40))
# a=torch.randn(10,10,10,40,3).cuda(0)
# a=spconv.SparseConvTensor.from_dense(a)
# conv=spconv.SubMConv3d(3,3,3)
# print(conv(a))
# model_para(SABlock(10,10),datasize=(1,10,10))
# model_para(unet.UNet(spatial_dims=3,in_channels=3,out_channels=1,channels=(6,12,24),strides=(1,1,1),num_res_units=3),datasize=[2,3,10,10,40],depth=5)
# model_para(MLP3_3D_Direc(),datasize=[3,10,10,40,3])
# model_para(MLP3(),datasize=[3,1,17,17])
# model_optim()
# model=MLP3()
# print(model._get_name())
# print(f'Total Time used: {time.time()-start}s')
# model_para(SparseCSPP(),datasize=(2,10,10,40,3))
model_para(STRM(),datasize=[2,3,10,10,40])
del_pycache()

# model_para(model=MLP3v2(),datasize=[3,1,17,17])
# logger=AirLogger.LOGT()
# logger.log(f'yes{1e4}')
# logger.log(f'{time.ctime()}')
