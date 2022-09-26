'''
Author: airscker
Date: 2022-09-20 23:29:14
LastEditors: airscker
LastEditTime: 2022-09-22 21:53:35
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import time
import os
import numpy as np
import argparse
import sys

# import AirFunc
from DeepMuon.models import *
from DeepMuon.dataset import *
# from config import *
# import AirLogger
import importlib

# import torch
from torch import nn

class Config:
    def __init__(self,configpath:str):
        '''
        ## Load Training Configuration from Python File stored in Folder 'config'
        - Args:
            - configpath: The path of the config file, must be in 'config' folder
        - Attributions:
            - paras: The parameters in config file, dtype: dict
                ```
                'model',
                'train_dataset',
                'test_dataset',
                'work_config',
                'checkpoint_config',
                'loss_fn',
                'hyperpara',
                'lr_config',
                'gpu_config'
                ```
        '''
        self.paras={'model':None,
                    'train_dataset':None,
                    'test_dataset':None,
                    'work_config':None,
                    'checkpoint_config':None,
                    'loss_fn':None,
                    'hyperpara':None,
                    'lr_config':None,
                    'gpu_config':None}
        self.config=self.__import_config(self,configpath=configpath)
        self.__check_config()
        self.__para_config()
        # print(self.paras)
    def __check_config(self):
        paras_config=dir(self.config)
        error=[]
        paras_check=list(self.paras.keys())
        for i in range(len(paras_check)):
            if paras_check[i] not in paras_config:
                error.append(paras_check)
        assert len(error)==0,f'These basic configurations are not specified in {self.configpath}:\n{error}'
    def __para_config(self):
        info=globals()
        self.paras['model']={'backbone':info[self.config.model['backbone']]}
        self.paras['train_dataset']={'backbone':info[self.config.train_dataset['backbone']],'datapath':self.config.train_dataset['datapath']}
        self.paras['test_dataset']={'backbone':info[self.config.test_dataset['backbone']],'datapath':self.config.test_dataset['datapath']}
        self.paras['work_config']=self.config.work_config
        self.paras['checkpoint_config']=self.config.checkpoint_config
        if self.config.loss_fn is None:
            self.paras['loss_fn']={'backbone':nn.MSELoss}
        else:
            self.paras['loss_fn']={'backbone':info[self.config.loss_fn['backbone']]}
        self.paras['hyperpara']=self.config.hyperpara
        self.paras['lr_config']=self.config.lr_config
        self.paras['gpu_config']=self.config.gpu_config
    
    @staticmethod
    def __import_config(self,configpath):
        assert '/' in configpath,f'Do not use standard windows path"\\", but {configpath} is given'
        assert configpath.endswith('.py'),f'Config file must be a python file but {configpath} is given'
        total_path=os.path.join(os.getcwd(),configpath.replace('./',''))
        assert os.path.exists(total_path),f'Configuration file {total_path} does not exist. Please check the path again'
        configpath=configpath.replace('.py','')
        importdirs=configpath.split('/')                
        sys.path.insert(0,configpath.replace(importdirs[-1],''))
        return importlib.import_module(importdirs[-1])
# config=Config(configpath='./DeepMuon/config/Hailing/MLP3_3D.py')
# env=globals()
# keys=list(env.keys())
# for i in range(len(keys)):
#     print(f'{keys[i]}: {env[keys[i]]}')
# print(__file__)