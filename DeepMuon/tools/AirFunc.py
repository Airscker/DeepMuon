'''
Author: Airscker
Date: 2022-09-02 14:37:59
LastEditors: airscker
LastEditTime: 2023-07-26 19:16:10
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import os
import shutil
import parso
import socket
import GPUtil
import importlib
import warnings
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from yapf.yapflib.yapf_api import FormatCode

import torch
from torch import nn

def check_port(ip:str='127.0.0.1',port:int=8080):
    '''check the socket port's availability'''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((ip,port))
    if result == 0:
        return False
    else:
        return True


def fix_port(ip:str='127.0.0.1',port:int=8080):
    '''check the socket port's availbility and find usable port'''
    new_port=port
    while True:
        port_usable=check_port(ip=ip,port=new_port)
        if not port_usable:
            new_port+=1
        else:
            break
    info=None
    if new_port!=port:
        info=f'WARN: Port {port} is unavailable, we reset it as the nearest usable port {new_port}'
    return new_port,info


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
    GPU_group=GPUtil.getGPUs()
    if gpu_id is None:
        gpu_id = torch.cuda.current_device()
    mem_total=GPU_group[gpu_id].memoryTotal
    mem_left=GPU_group[gpu_id].memoryFree
    mem_used=GPU_group[gpu_id].memoryUsed
    return dict(mem_left=f'{mem_left} MB',
                mem_used=f'{mem_used} MB',
                total_mem=f'{mem_total} MB')


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

def load_json_log(log_file: str, start_from: int = 0,unique_key:int=None):
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
    if not log_file.endswith('.json'):
        warnings.warn(f"log_file must be json file, however, {log_file} given")
        return {}
    if not os.path.exists(log_file):
        warnings.warn(f'Training log {log_file} can not be found')
        return {}
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

def import_module(module_path: str) -> importlib.types.ModuleType:
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

def generate_nnhs_config(path:str=None,save_path:str=None,new_params:dict=None):
    '''
    ## Generate neural network hyperparameter searching configuartion for every trail
    
    ### Args:
        - path: specify the overall configuartion path of NNHS experiments
        - save_path: specify the path to save the NNHS trail configuration
        - new_params: specify the new parameters given by NNHS search space
    
    ### Returns:
        - new_config: the new module which contains hyperparameters refreshed
        - new_source: the source code of generated NNHS configuration
    '''
    config_info=import_module(path)
    target=parse_config(path=path,key='search_params',formatted=False)
    source=module_source(module_path=path)
    config_info.search_params.update(new_params)
    source=source.replace(target,f"search_params={config_info.search_params}\n")
    try:
        source=FormatCode(source)[0]
    except:
        print('Unexpected error occured in formatting NNHS configuration file, unformatted file was saved.')
    tmp_path=path.replace('.py',f'_{os.getpid()}.py')
    with open(tmp_path,'w+')as f:
        f.write(source)
    f.close()
    new_config=import_module(tmp_path)
    config_elements=dir(new_config)
    if save_path is not None:
        with open(save_path,'w+')as f:
            f.write(source)
        f.close()
    os.remove(tmp_path)
    return new_config,source



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
