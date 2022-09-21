'''
Author: Airscker
Date: 2022-09-02 14:37:59
LastEditors: airscker
LastEditTime: 2022-09-21 23:10:50
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import time
import os
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import click
import numpy as np
import seaborn as sns


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.utils.data import DataLoader
from torchsummary import summary
# from monai.networks.blocks import 
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)

class PandaxTensorData():
    def __init__(self,IMG_XY_path='..\\data\\IMG2D_XY.pkl'):
        self.IMG_XY_path=IMG_XY_path
        self.IMGs=[]
        self.labels=[]
        self.__Init()
    def gettensor(self, idx):
        image=self.IMGs[idx]
        image=np.reshape(image,(1,image.shape[0],image.shape[1]))
        label=self.labels[idx]/100# Key action
        image=torch.from_numpy(image)
        label=torch.from_numpy(label)
        return image, label
    def getitem(self,idx):
        return self.IMGs[idx],self.labels[idx]/100
    def __Init(self):
        data=pkl.load(open(self.IMG_XY_path,'rb'))
        img=[]
        label=[]
        for i in range(len(data)):
            img.append(data[i][0])
            label.append(data[i][1])
        self.IMGs=np.array(img)
        self.labels=np.array(label)
        return img,label

def hist_plot(data,inter=20,xlabel='The number of events'):
    data=np.array(data)
    plt.figure(figsize=(15,8))
    plt.xlabel(xlabel)
    plt.ylabel('Frequency Count')
    plt.title(f'Total number of events: {len(data)}\
        \nMAX/MIN: [{np.max(data)}][{np.min(data)}]\
        \nAverage Value: {np.mean(data)}\
        \nWidth of bin: {inter}')
    bins=list(np.arange(np.min(data)-inter,np.max(data)+inter,inter))
    plt.xticks(bins)
    n,bins,_=plt.hist(data,rwidth=0.9,bins=bins)
    for i in range(len(n)):
        plt.text(bins[i], n[i]*1.02, round(n[i],6), fontsize=12, horizontalalignment="left")
    plt.show()
    return 0

def format_time(second):
    '''Get formatted time: H:M:S'''
    hours=int(second//3600)
    second-=3600*hours
    minutes=int(second//60)
    second=int(second%60)
    return f'{hours}:{minutes:02d}:{second:02d}'

def save_model(epoch:int,model:nn.Module,optimizer,loss_fn,schedular,path,dist_train=False):
    """Save a model to disk, Only their state_dict()
    Args:
        epoch\n
        model\n
        optimizer\n
        loss_fn\n
        schedular\n
        path\n
        dist_train\n
    """
    torch.save({
            'epoch': epoch,
            'model': model.state_dict() if dist_train==False else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_fn': loss_fn.state_dict(),
            'schedular':schedular.state_dict(),
            }, path)
    return 0
def load_model(path,device):
    """Loads the model and optimizer parameters from a previously saved checkpoint file .
    Args:
        path: The checkpoint path
    Returns:
        epoch: last trained epoch\n
        model_dic\n
        optimizer_dic\n
        schedular_dic\n
        loss_fn_dic\n
    """
    checkpoint = torch.load(path,map_location=device)
    model_dic=checkpoint['model']
    optimizer_dic=checkpoint['optimizer']
    schedular_dic=checkpoint['schedular']
    epoch = checkpoint['epoch']
    loss_fn_dic=checkpoint['loss_fn']
    return epoch,model_dic,optimizer_dic,schedular_dic,loss_fn_dic

# print(__file__)