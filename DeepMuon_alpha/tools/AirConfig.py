'''
Author: airscker
Date: 2022-09-20 23:29:14
LastEditors: airscker
LastEditTime: 2022-10-30 13:02:55
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
        self.config=self.__import_config(configpath=configpath)
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
        '''imoprt the model anywhere'''
        model_info=getattr(self.config,'model')
        if 'params' not in model_info.keys():
            model_params={}
        else:
            model_params=model_info['params']
        if 'filepath' not in model_info.keys() or not os.path.exists(model_info['filepath']):
            self.paras['model']={'backbone':info[self.config.model['backbone']],'params':model_params}
        else:
            self.paras['model']={'backbone':getattr(self.__import_config(model_info['filepath']),model_info['backbone']),'params':model_params}
        # self.paras['model']={'backbone':info[self.config.model['backbone']]}
        '''import dataset anywhere'''
        traindataset_info=getattr(self.config,'train_dataset')
        testdataset_info=getattr(self.config,'test_dataset')
        if 'filepath' not in traindataset_info.keys() or not os.path.exists(traindataset_info['filepath']):
            if 'params' not in traindataset_info:
                self.paras['train_dataset']={'backbone':info[self.config.train_dataset['backbone']],
                                            'params':dict(datapath=self.config.train_dataset['datapath'])}
            else:
                self.paras['train_dataset']={'backbone':info[self.config.train_dataset['backbone']],
                                            'params':traindataset_info['params']}
        else:
            if 'params' not in traindataset_info:
                self.paras['train_dataset']={'backbone':getattr(self.__import_config(traindataset_info['filepath']),traindataset_info['backbone']),
                                            'params':dict(datapath=self.config.train_dataset['datapath'])}
            else:
                self.paras['train_dataset']={'backbone':getattr(self.__import_config(traindataset_info['filepath']),traindataset_info['backbone']),
                                            'params':traindataset_info['params']}
        if 'filepath' not in testdataset_info.keys() or not os.path.exists(testdataset_info['filepath']):
            if 'params' not in testdataset_info:
                self.paras['test_dataset']={'backbone':info[self.config.test_dataset['backbone']],
                                            'params':dict(datapath=self.config.test_dataset['datapath'])}
            else:
                self.paras['test_dataset']={'backbone':info[self.config.test_dataset['backbone']],
                                            'params':testdataset_info['params']}
        else:
            if 'params' not in testdataset_info:
                self.paras['test_dataset']={'backbone':getattr(self.__import_config(testdataset_info['filepath']),testdataset_info['backbone']),
                                            'params':dict(datapath=self.config.test_dataset['datapath'])}
            else:
                self.paras['test_dataset']={'backbone':getattr(self.__import_config(testdataset_info['filepath']),testdataset_info['backbone']),
                                            'params':testdataset_info['params']}
        # self.paras['train_dataset']={'backbone':info[self.config.train_dataset['backbone']],'datapath':self.config.train_dataset['datapath']}
        # self.paras['test_dataset']={'backbone':info[self.config.test_dataset['backbone']],'datapath':self.config.test_dataset['datapath']}
        '''import loss function anywhere'''
        if self.config.loss_fn is None:
            self.paras['loss_fn']={'backbone':nn.MSELoss,'params':dict()}
        else:
            loss_info=getattr(self.config,'loss_fn')
            if 'filepath' not in loss_info.keys() or not os.path.exists(loss_info['filepath']):
                if 'params' not in loss_info.keys():
                    loss_fn_para=dict()
                else:
                    loss_fn_para=loss_info['params']
                self.paras['loss_fn']={'backbone':info[self.config.loss_fn['backbone']],'params':loss_fn_para}
            else:
                if 'params' not in loss_info.keys():
                    loss_fn_para=dict()
                else:
                    loss_fn_para=loss_info['params']
                self.paras['loss_fn']={'backbone':getattr(self.__import_config(loss_info['filepath']),loss_info['backbone']),'params':loss_fn_para}
        self.paras['work_config']=self.config.work_config
        self.paras['checkpoint_config']=self.config.checkpoint_config
        self.paras['hyperpara']=self.config.hyperpara
        self.paras['lr_config']=self.config.lr_config
        self.paras['gpu_config']=self.config.gpu_config
    @staticmethod
    def __import_config(configpath):
        assert configpath.endswith('.py'),f'Config file must be a python file but {configpath} is given'
        total_path=os.path.join(os.getcwd(),configpath.replace('./',''))
        assert os.path.exists(total_path),f'Configuration file {total_path} does not exist. Please check the path again'
        total_path=total_path.replace(os.getcwd(),'').replace('\\','/')
        if total_path.startswith('/'):
            total_path=total_path[1:]
        total_path=total_path.replace('/','.').replace('.py','')
        return importlib.import_module(total_path)
# config=Config(configpath='./DeepMuon/config/Hailing/CSPP.py')
# env=globals()
# keys=list(env.keys())
# for i in range(len(keys)):
#     print(f'{keys[i]}: {env[keys[i]]}')
# print(__file__)