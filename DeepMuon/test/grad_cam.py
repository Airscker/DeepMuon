'''
Author: airscker
Date: 2022-10-16 18:08:59
LastEditors: airscker
LastEditTime: 2022-10-16 18:10:29
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import torch
from torch import nn
from captum.attr import IntegratedGradients

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