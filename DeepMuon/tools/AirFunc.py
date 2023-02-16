'''
Author: Airscker
Date: 2022-09-02 14:37:59
LastEditors: airscker
LastEditTime: 2023-02-16 19:01:53
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import os
import shutil
import importlib
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import nn
torch.set_default_tensor_type(torch.DoubleTensor)


def exclude_key(dictionary: dict, del_key: str = 'type'):
    '''
    Delete key-value map from dictionary
    '''
    new_dict = {}
    for key in dictionary.keys():
        if key != del_key:
            new_dict[key] = dictionary[key]
    return new_dict


def get_mem_info(gpu_id=None):
    '''
    ## Get the memory information of specified GPU

    ### Args:
        - gpu_id: the id of GPU

    ### Return:
        - dict:
            - mem_left: the memory unused(in MB format)
            - mem_used: the meory used(in MB format)
            - total_mem: the total memory of GPU(in MB format)
    '''
    if gpu_id is None:
        gpu_id = torch.cuda.current_device()
    mem_total = torch.cuda.get_device_properties(gpu_id).total_memory
    mem_cached = torch.cuda.memory_reserved(gpu_id)
    mem_allocated = torch.cuda.memory_allocated(gpu_id)
    return dict(mem_left=f"{(mem_total-mem_cached-mem_allocated)/1024**2:0.2f} MB",
                mem_used=f"{(mem_cached+mem_allocated)/1024**2:0.2f} MB",
                total_mem=f"{mem_total/1024**2:0.2f} MB")


def readable_dict(data: dict, i=0, show=False, indent='\t', sep='\n'):
    """
    ## Convert a dictionary to a more readable format.

    ### Args:
        - data: Pass the data to be printed
        - i: Control the indentation of the output
        - show: Whether to print out the mesage in console
        - indent: the indent letter used to convert dictionary
        - spe: the seperation letter used to seperate dict elements

    ### Return:
        - A string represent the dictionary
    """
    info = ''
    for key in data:
        info += indent*i
        info += f'{key}: '
        if isinstance(data[key], dict):
            info += f"{sep}{readable_dict(data[key], i+1,indent=indent,sep=sep)}"
        else:
            info += f"{data[key]}{sep}"
    if show:
        print(info)
    return info


def unpack_json_log(log_path: str, start_from: int = 0) -> list:
    """
    ## Unpack data in json log file

    ### Args:
        - log_path: the path of logfile
        - start_from: the index of line to start with

    ### Return:
        - list(dict()): the list of extracted data dictionaries
    """
    with open(log_path, 'r')as f:
        ori_data = f.readlines()
    ori_data = ori_data[start_from:]
    info = []
    for i in range(len(ori_data)):
        try:
            data = eval(ori_data[i].split('\n')[0])
        except:
            pass
        if 'mode' in data.keys():
            info.append(data)
    return info


def load_json_log(log_file: str, start_from: int = 0) -> np.ndarray:
    '''
    ## Load json data from json logfile

    ### Args:
        - log_file: the path of json logfile
        - start_from: the index of line to start with

    ### Return:
        - dict(list()):
    '''
    assert log_file.endswith(
        '.json'), f"log_file must be json file, however, {log_file} given"
    assert os.path.exists(
        log_file), f'Training log {log_file} can not be found'
    json_log = unpack_json_log(log_file, start_from)
    log_info = {}
    for i in range(len(json_log)):
        if json_log[i]['mode'] in log_info.keys():
            log_info[json_log[i]['mode']].append(
                list(exclude_key(json_log[i], del_key='mode').values()))
        else:
            log_info[json_log[i]['mode']] = [
                list(exclude_key(json_log[i], del_key='mode').values())]
    return log_info


def load_log(log_file: str) -> np.ndarray:
    """
    ## Loads the training log from the given log file .

    ### Args:
        - log_file ([type]): The path of the log file.

    ### Return:
        - np.array: loaded training results, with shape [epoch:[lr,tsl,trl,btsl]]

    """
    assert os.path.exists(
        log_file), f'Training log {log_file} can not be found'
    with open(log_file, 'r')as f:
        info = f.readlines()
    train_info = []
    for i in range(len(info)):
        info[i] = info[i].split('\n')[0]
        if 'LR' in info[i] and 'Test Loss' in info[i] and 'Train Loss' in info[i] and 'Best Test Loss' in info[i]:
            data = info[i].split(',')
            epoch = int(data[1].split('[')[1].split(']')[0])
            train_data = [float(data[0].split(': ')[-1]), float(data[2].split(': ')[-1]),
                          float(data[3].split(': ')[-1]), float(data[4].split(': ')[-1])]
            if epoch > len(train_info):
                train_info.append(train_data)
            else:
                train_info[epoch-1] = train_data
    return np.array(train_info)


def import_module(module_path: str):
    '''
    ## Import python module according to the file path

    ### Args:
        - module_path: the path of the module to be imported

    ### Return:
        - the imported module
    '''
    # assert '/' in module_path,f'Do not use standard windows path"\\", but {module_path} is given'
    assert module_path.endswith(
        '.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.join(os.getcwd(), module_path.replace('./', ''))
    assert os.path.exists(
        total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def plot_hist_2nd(data, title='x', bins=15, sigma=3, save='', show=False):
    """
    ## Plots a histogram of the data provided.
    It also includes lines to represent the mean and +/- 3 standard deviations.

    ### Args:
        - data: The list of numbers that will be plotted as a histogram
        - title: The title for the plot
        - bins: The number of bins to use for the histogram (default is 15).
        - sigma: How many standard deviations away from mean should be highlighted (default is 3).
        - save: The path to save the ploted image, if '' given, saving action will be canceled
        - show: Whether to show the ploted image within console
    """
    plt.figure(figsize=(20, 8))
    plt.title(f'Distribution of {title} Total Number: {len(data)}\
        \nMIN/MAX: {np.min(data)}/{np.max(data)} MEAN/STD: {np.mean(data)}/{np.std(data)}\
        \n{sigma}Sigma: {np.mean(data)+sigma*np.std(data)} {np.mean(data)-sigma*np.std(data)}')
    # n,bins,patchs=plt.hist(data,bins=list(np.arange(np.min(data)-0.01,np.max(data)+0.01,0.01)),rwidth=0.9)
    n, bins, patchs = plt.hist(data, bins=bins, rwidth=0.9)
    for i in range(len(n)):
        plt.text(bins[i], n[i]*1.02, round(n[i], 6),
                 fontsize=12, horizontalalignment="left")
    sigma_rangex = np.array(
        (np.mean(data)-sigma*np.std(data), np.mean(data)+sigma*np.std(data)))
    axis = plt.axis()
    plt.text(x=np.mean(sigma_rangex),
             y=axis[-1]/1.2, s=f'+/- {sigma} Sigma Range', ha='center')
    plt.text(x=np.mean([np.min(data), sigma_rangex[0]]), y=axis[-1]/2,
             s=f'Sample Number: {np.count_nonzero(data<sigma_rangex[0])}', ha='center')
    plt.text(x=np.mean([sigma_rangex[1], np.max(data)]), y=axis[-1]/2,
             s=f'Sample Number: {np.count_nonzero(data>sigma_rangex[1])}', ha='center')
    plt.fill_betweenx(y=axis[-2:], x1=max(np.min(data), sigma_rangex[0]),
                      x2=min(np.max(data), sigma_rangex[1]), alpha=0.2)
    plt.fill_betweenx(y=np.array(
        axis[-2:])/2, x1=np.min(data), x2=sigma_rangex[0], alpha=0.2)
    plt.fill_betweenx(y=np.array(
        axis[-2:])/2, x1=sigma_rangex[1], x2=np.max(data), alpha=0.2)
    plt.axis(axis)
    if save != '':
        plt.savefig(save)
    if show == True:
        plt.show()


def plot_curve(data, title='Curve', axis_label=['Epoch', 'Loss'], data_label=['Curve1'], save='', mod='min', show=False):
    """
    ## Plots a single or multiple curves on the same plot.
    The function takes in a list of data and labels for each curve to be plotted.
    The axis labels are optional, but if provided they will be used as x-axis label and y-axis label respectively.
    If no axis labels are provided, then the default values `Epoch` and `Loss&quot` will be used instead.

    ### Args:
        - data: Plot the data, it can be a list of numpy array or a single numpy array
        - title: Set the title of the plot
        - axis_label: Set the labels of the every axis
        - data_label: Labels of the curves
        - save: The path to save the ploted image, if '' given, saving action will be canceled
        - mod: plot the max/min value
        - show: Whether to show the ploted image within console
    """
    # data=np.array(data)
    plt.figure(figsize=(20, 10))
    plt.title(f'{title}')
    assert mod == 'min' or mod == 'max', f"mod must be 'min'/'max', however mod='{mod}' given"
    if isinstance(data[0], list) or isinstance(data[0], np.ndarray):
        for i in range(len(data)):
            data[i] = np.array(data[i])
            label = data_label[i] if len(data_label) >= i+1 else f'Curve{i+1}'
            if mod == 'min':
                label = f'{label} MIN/POS: {np.min(data[i])}/{np.argwhere(data[i]==np.min(data[i]))[-1]}'
                plt.axhline(np.min(data[i]), linestyle='-.')
            elif mod == 'max':
                label = f'{label} MAX/POS: {np.max(data[i])}/{np.argwhere(data[i]==np.max(data[i]))[-1]}'
                plt.axhline(np.max(data[i]), linestyle='-.')
            plt.plot(data[i], label=label)

    else:
        data = np.array(data)
        label = data_label[0] if len(data_label) >= 1 else f'Curve1'
        if mod == 'min':
            label = f'{label} MIN/POS: {np.min(data)}/{np.argwhere(data==np.min(data))[-1]}'
            plt.axhline(np.min(data), linestyle='-.')
        elif mod == 'max':
            label = f'{label} MAX/POS: {np.max(data)}/{np.argwhere(data==np.max(data))[-1]}'
            plt.axhline(np.max(data), linestyle='-.')
        plt.plot(data, label=label)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.grid()
    plt.legend()
    if save != '':
        plt.savefig(save, dpi=400)
    if show:
        plt.show()


def format_time(second):
    '''Get formatted time: H:M:S'''
    hours = int(second//3600)
    second -= 3600*hours
    minutes = int(second//60)
    second = int(second % 60)
    return f'{hours}:{minutes:02d}:{second:02d}'


def save_model(epoch: int, model: nn.Module, optimizer, loss_fn, scheduler, path, dist_train=False):
    """
    ## Save a model to disk, Only their state_dict()

    ### Args:
        - epoch
        - model
        - optimizer
        - loss_fn
        - scheduler
        - path
        - dist_train
    """
    torch.save({
        'epoch': epoch,
        'model': model.state_dict() if dist_train == False else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, path)


def load_model(path: str, device: torch.device):
    """
    ## Loads the model and optimizer parameters from a previously saved checkpoint file.

    ### Args:
        - path: The checkpoint path

    ### Returns:
        - epoch: last trained epoch
        - model_dic
        - optimizer_dic
        - scheduler_dic
        - loss_fn_dic
    """
    checkpoint = torch.load(path, map_location=device)
    model_dic = checkpoint['model']
    optimizer_dic = checkpoint['optimizer']
    try:
        scheduler_dic = checkpoint['scheduler']
    except:
        scheduler_dic = checkpoint['schedular']
    epoch = checkpoint['epoch']
    loss_fn_dic = checkpoint['loss_fn']
    return epoch, model_dic, optimizer_dic, scheduler_dic, loss_fn_dic


def del_pycache(path='./'):
    """
    ## Delete all the python cache files in a directory

    ### Args:
        - path: the root path of workspace. Defaults to './'.

    ### Returns:
        - the list of all deleted cache folders' path
    """
    cache = []
    for root, dirs, files in os.walk(path):
        # print(root,dirs,files)
        if root.endswith('__pycache__'):
            shutil.rmtree(root)
            cache.append(root)
            print(f'{root} was deleted')
    return cache
