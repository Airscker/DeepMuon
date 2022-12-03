'''
Author: Airscker
Date: 2022-09-02 14:37:59
LastEditors: airscker
LastEditTime: 2022-12-03 23:31:18
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import time
import os
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import click
import sys
import numpy as np
import shutil
import importlib


import torch
from torch import nn
# from torch import Tensor
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
# import torchvision.models as models
# from torch.utils.data import DataLoader
# from monai.networks.blocks import 
# from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)


def load_log(log_file):
    """Loads the training log from the given log file .

    Args:
        log_file ([type]): The path of the log file.
    
    Return:
        [epoch:[lr,tsl,trl,btsl]]

    """
    assert os.path.exists(log_file),f'Training log {log_file} can not be found'
    with open(log_file,'r')as f:
        info=f.readlines()
    train_info=[]
    for i in range(len(info)):
        info[i]=info[i].split('\n')[0]
        if 'LR' in info[i] and 'Test Loss' in info[i] and 'Train Loss' in info[i] and 'Best Test Loss' in info[i]:
            data=info[i].split(',')
            epoch=int(data[1].split('[')[1].split(']')[0])
            train_data=[float(data[0].split(': ')[-1]),float(data[2].split(': ')[-1]),float(data[3].split(': ')[-1]),float(data[4].split(': ')[-1])]
            if epoch>len(train_info):
                train_info.append(train_data)
            else:
                train_info[epoch-1]=train_data
    return np.array(train_info)

def import_module(module_path):
    # assert '/' in module_path,f'Do not use standard windows path"\\", but {module_path} is given'
    assert module_path.endswith('.py'),f'Config file must be a python file but {module_path} is given'
    total_path=os.path.join(os.getcwd(),module_path.replace('./',''))
    assert os.path.exists(total_path),f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('',total_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

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

def plot_hist_2nd(data,title='x',bins=15,sigma=3,save='',show=False):
    """
    The plot_hist_2nd function plots a histogram of the data provided. 
    It also includes lines to represent the mean and +/- 3 standard deviations.
    The function takes in 4 parameters: 
        1) data - The list of numbers that will be plotted as a histogram
        2) title - The title for the plot
        3) bins - The number of bins to use for the histogram (default is 15).
        4) sigma - How many standard deviations away from mean should be highlighted (default is 3).
    
    :param data: Plot the histogram
    :param title='x': Set the title of the plot
    :param bins=15: Set the number of bins in the histogram
    :param sigma=3: Set the sigma range of the distribution
    :param save='': Save the plot as a 
    :param show=False: Save the plot without showing it
    :return: The n, bins, patchs from the plt
    """
    plt.figure(figsize=(20,8))
    plt.title(f'Distribution of {title} Total Number: {len(data)}\
        \nMIN/MAX: {np.min(data)}/{np.max(data)} MEAN/STD: {np.mean(data)}/{np.std(data)}\
        \n{sigma}Sigma: {np.mean(data)+sigma*np.std(data)} {np.mean(data)-sigma*np.std(data)}')
    # n,bins,patchs=plt.hist(data,bins=list(np.arange(np.min(data)-0.01,np.max(data)+0.01,0.01)),rwidth=0.9)
    n,bins,patchs=plt.hist(data,bins=bins,rwidth=0.9)
    for i in range(len(n)):
        plt.text(bins[i], n[i]*1.02, round(n[i],6), fontsize=12, horizontalalignment="left")
    sigma_rangex=np.array((np.mean(data)-sigma*np.std(data),np.mean(data)+sigma*np.std(data)))
    axis=plt.axis()
    plt.text(x=np.mean(sigma_rangex),y=axis[-1]/1.2,s=f'+/- {sigma} Sigma Range',ha='center')
    plt.text(x=np.mean([np.min(data),sigma_rangex[0]]),y=axis[-1]/2,s=f'Sample Number: {np.count_nonzero(data<sigma_rangex[0])}',ha='center')
    plt.text(x=np.mean([sigma_rangex[1],np.max(data)]),y=axis[-1]/2,s=f'Sample Number: {np.count_nonzero(data>sigma_rangex[1])}',ha='center')
    plt.fill_betweenx(y=axis[-2:],x1=max(np.min(data),sigma_rangex[0]),x2=min(np.max(data),sigma_rangex[1]),alpha=0.2)
    plt.fill_betweenx(y=np.array(axis[-2:])/2,x1=np.min(data),x2=sigma_rangex[0],alpha=0.2)
    plt.fill_betweenx(y=np.array(axis[-2:])/2,x1=sigma_rangex[1],x2=np.max(data),alpha=0.2)
    plt.axis(axis)
    if save!='':
        plt.savefig(save)
    if show==True:
        plt.show()
def plot_curve(data,title='Curve',axis_label=['Epoch','Loss'],data_label=['Curve1'],save='',show=False):
    """
    The plot_curve function plots a single or multiple curves on the same plot.
    The function takes in a list of data and labels for each curve to be plotted.
    The axis labels are optional, but if provided they will be used as x-axis label and y-axis label respectively. 
    If no axis labels are provided, then the default values &quot;Epoch&quot; and &quot;Loss&quot; will be used instead.
    
    :param data: Plot the data, it can be a list of numpy array or a single numpy array
    :param title='Curve': Set the title of the plot
    :param axis_label=['Epoch': Set the label of the x-axis
    :param 'Loss']: Set the title of the graph
    :param data_label=['Curve1']: Label the curve in the plot
    :param save='': Save the plot to a file
    :param show=False: Save the plot as an image file
    :return: Nothing, it just plots the curve on the current figure
    """
    # data=np.array(data)
    plt.figure(figsize=(20,10))
    plt.title(f'{title}')
    if isinstance(data[0],list) or isinstance(data[0],np.ndarray):
        for i in range(len(data)):
            data[i]=np.array(data[i])
            label=data_label[i] if len(data_label)>=i+1 else f'Curve{i+1}'
            plt.plot(data[i],label=f'{label} MIN: {np.min(data[i])}')
            plt.axhline(np.min(data[i]),linestyle='-.')
    else:
        data=np.array(data)
        label=data_label[0] if len(data_label)>=1 else f'Curve1'
        plt.plot(data,label=f'{label} MIN: {np.min(data)}')
        plt.axhline(np.min(data),linestyle='-.')
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.grid()
    plt.legend()
    if save!='':
        plt.savefig(save)
    if show:
        plt.show()
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
def load_model(path:str,device:torch.device):
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
def del_pycache(path='./'):
    """Delete all the python cache files in a directory

    Args:
        path (str, optional): the root path of workspace. Defaults to './'.

    Returns:
        cache: the list of all deleted cache folders' path
    """
    cache=[]
    for root,dirs,files in os.walk(path):
        # print(root,dirs,files)
        if root.endswith('__pycache__'):
            shutil.rmtree(root)
            cache.append(root)
            print(f'{root} was deleted')
    return cache