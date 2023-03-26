'''
Author: Airscker
Date: 2022-09-02 14:37:59
LastEditors: airscker
LastEditTime: 2023-03-26 23:15:47
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import os
import shutil
import parso
from yapf.yapflib.yapf_api import FormatCode
import importlib
import warnings
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


import torch
from torch import nn


def check_device(device: Union[int, str, torch.device]):
    '''check the cuda/cpu device specified'''
    if not torch.cuda.is_available():
        warnings.warn(f"CUDA is not available, device is replaced as 'cpu'")
        device = 'cpu'
    elif isinstance(device, int):
        if device+1 > torch.cuda.device_count():
            warnings.warn(
                f"Only {torch.cuda.device_count()} devices available, however cuda:{device} is specified. We will use cuda:{torch.cuda.device_count()-1} instead")
        device = torch.device(min(device, torch.cuda.device_count()-1))
    elif isinstance(device, torch.device):
        if device.index+1 > torch.cuda.device_count():
            warnings.warn(
                f"Only {torch.cuda.device_count()} devices available, however {device} is specified. We will use cuda:{torch.cuda.device_count()-1} instead")
            device = torch.device(torch.cuda.device_count()-1)
    return device


def plot_3d(img, save='', show=False, title='', norm=False,vector=None):
    """
    ## Plot the 3D image of the given image.

    ### Args:
        - img: the image data to plot
        - save: the path to save the image
        - show: whether to show the image
        - title: the title of the image
        - norm: whether to normalize the image
        - vector: (optional)the vector to be ploted, [x0,y0,z0,nx,ny,nz]
    """
    x = []
    y = []
    z = []
    num = []
    img = np.array(img)
    if norm:
        img = (img-np.min(img))/(np.max(img)-np.min(img))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i][j][k] != 0:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    num.append(img[i][j][k])
    fig = plt.figure(figsize=(15, 15))
    plt.title(title)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=num, cmap='jet')
    if vector is not None:
        start_point=vector[:3]
        direction=vector[3:]
        direction=direction/np.sqrt(np.sum(direction**2))
        end_point=direction*5+start_point
        ax.quiver(start_point[0],start_point[1],start_point[2],end_point[0],end_point[1],end_point[2],color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save != '':
        plt.savefig(save, dpi=600)
    if show:
        plt.show()
    plt.clf()
    return 0


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


def load_json_log(log_file: str, start_from: int = 0,unique_key:int=None) -> np.ndarray:
    '''
    ## Load json data from json logfile

    ### Args:
        - log_file: the path of json logfile
        - start_from: the index of line to start with
        - unique_key: the key used to abandon the repeated information, such as epochs with the same index,
            set unique_key='epoch' will omit the former presented repeated epochs' information

    ### Return:
        - dict(dict(list())): {mode:{metric:[data]}}
    '''
    assert log_file.endswith(
        '.json'), f"log_file must be json file, however, {log_file} given"
    assert os.path.exists(
        log_file), f'Training log {log_file} can not be found'
    json_log = unpack_json_log(log_file, start_from)
    log_info = {}
    for i in range(len(json_log)):
        if unique_key in json_log[i].keys():
            unique_key_check=json_log[i][unique_key]
        else:
            unique_key_check=i
        if json_log[i]['mode'] not in log_info.keys():
            log_info[json_log[i]['mode']] = {unique_key_check:exclude_key(json_log[i], del_key='mode')}
        else:
            log_info[json_log[i]['mode']][unique_key_check]=exclude_key(json_log[i], del_key='mode')
    for mod in log_info.keys():
        new_info={}
        for key_checked in log_info[mod].keys():
            for info_key in log_info[mod][key_checked].keys():
                if info_key not in new_info.keys():
                    new_info[info_key]=[log_info[mod][key_checked][info_key]]
                else:
                    new_info[info_key].append(log_info[mod][key_checked][info_key])
        log_info[mod]=new_info
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
    assert module_path.endswith(
        '.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.abspath(module_path)
    assert os.path.exists(
        total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

def module_source(module_path: str):
    '''
    ## Get python module's source code according to the file path

    ### Args:
        - module_path: the path of the module to be imported

    ### Return:
        - the source code of the module
    '''
    assert module_path.endswith(
        '.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.abspath(module_path)
    assert os.path.exists(
        total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module_loader = module_spec.loader
    return module_loader.get_source('')


def parse_config(source:str=None,path:str=None,key:str=None,formatted=True):
    """
    ## This function parses a configuration file and returns the source code of a specific element.

    ### Args:
        - source (str): the source code of the configuration file
        - path (str): the path of the configuration file
        - key (str): the name of the element to be parsed
        - formatted (bool): whether or not to format the source code
        
    ### Return:
        - (str): the source code of the parsed element
    """
    assert source!=None or path!=None,f"At least one parameter of source / path expected."
    assert key is not None,f"Name of the parsed element expected."
    if source is None:
        source=module_source(path)
    parsed_code=parso.parse(source)
    target_source=None
    for child in parsed_code.children:
        if not isinstance(child,parso.python.tree.EndMarker):
            element_ins=child.children[0]
            if isinstance(element_ins,parso.python.tree.ExprStmt):
                element_source=child.get_code()
                if element_source.startswith(key):
                    target_source=element_source
    if target_source is not None and formatted:
        target_source=FormatCode(target_source)[0].rstrip('\n')
    return target_source

def generate_nni_config(path:str=None,save_path:str=None,new_params:dict=None):
    config_info=import_module(path)
    target=parse_config(path=path,key='search_params',formatted=False)
    source=module_source(module_path=path)
    config_info.search_params.update(new_params)
    source=source.replace(target,f"search_params={config_info.search_params}\n")
    source=FormatCode(source)[0]
    tmp_path=path.replace('.py','_tmp.py')
    with open(tmp_path,'w+')as f:
        f.write(source)
    f.close()
    new_config=import_module(tmp_path)
    config_elements=dir(new_config)
    new_source=''
    for key in config_elements:
        if key!='search_params' and not key.startswith('__'):
            new_source+=f"{key}={getattr(new_config,key)}\n"
    new_source=FormatCode(new_source)[0]
    if save_path is not None:
        with open(save_path,'w+')as f:
            f.write(new_source)
        f.close()
    os.remove(tmp_path)
    return new_config,new_source


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
            else:
                continue
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
        else:
            pass
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
    try:
        model_dic = checkpoint['model']
    except:
        model_dic=checkpoint
    try:
        optimizer_dic = checkpoint['optimizer']
    except:
        optimizer_dic=None
    try:
        scheduler_dic = checkpoint['scheduler']
    except:
        scheduler_dic = None
    try:
        epoch = checkpoint['epoch']
    except:
        epoch=0
    try:
        loss_fn_dic = checkpoint['loss_fn']
    except:
        loss_fn_dic=None
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
