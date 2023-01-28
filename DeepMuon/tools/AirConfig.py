'''
Author: airscker
Date: 2022-09-20 23:29:14
LastEditors: airscker
LastEditTime: 2023-01-28 15:09:15
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import os
from DeepMuon.models import *
from DeepMuon.dataset import *
from DeepMuon.tools.AirFunc import import_module
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn.modules.loss import *
import warnings


class Config:
    def __init__(self, configpath: str):
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
                'optimizer',
                'schedular',
                'gpu_config'
                ```
        '''
        self.paras = {'model': None,
                      'train_dataset': None,
                      'test_dataset': None,
                      'work_config': None,
                      'checkpoint_config': None,
                      'loss_fn': None,
                      'hyperpara': None,
                      'optimizer': None,
                      'schedular': None,
                      'lr_config': None,
                      'gpu_config': None}
        self.config = import_module(configpath)
        self.config_keys = dir(self.config)
        self.configpath = configpath
        self.__check_config()
        self.__para_config()
        # print(self.paras)

    def __check_config(self):
        paras_config = self.config_keys
        error = []
        paras_check = list(self.paras.keys())
        for i in range(len(paras_check)):
            if paras_check[i] not in paras_config:
                error.append(paras_check[i])
        if 'lr_config' in error and 'optimizer' not in error and 'schedular' not in error:
            error.remove('lr_config')
        if 'lr_config' not in error:
            if 'schedular' in error:
                error.remove('schedular')
            if 'optimizer' in error:
                error.remove('optimizer')
        assert len(
            error) == 0, f'These basic configurations are not specified in {self.configpath}:\n{error}'

    def __para_config(self):
        internal_env = globals()
        '''import the model anywhere'''
        model_info = getattr(self.config, 'model')
        if 'params' not in model_info.keys():
            model_params = {}
        else:
            model_params = model_info['params']
        if 'filepath' not in model_info.keys() or not os.path.exists(model_info['filepath']):
            self.paras['model'] = {'backbone': internal_env[self.config.model['backbone']],
                                   'params': model_params}
        else:
            self.paras['model'] = {'backbone': getattr(import_module(model_info['filepath']), model_info['backbone']),
                                   'params': model_params}
        # self.paras['model']={'backbone':internal_env[self.config.model['backbone']]}
        '''import dataset anywhere'''
        traindataset_info = getattr(self.config, 'train_dataset')
        testdataset_info = getattr(self.config, 'test_dataset')
        if 'filepath' not in traindataset_info.keys() or not os.path.exists(traindataset_info['filepath']):
            if 'params' not in traindataset_info:
                self.paras['train_dataset'] = {'backbone': internal_env[self.config.train_dataset['backbone']],
                                               'params': dict(datapath=self.config.train_dataset['datapath'])}
            else:
                self.paras['train_dataset'] = {'backbone': internal_env[self.config.train_dataset['backbone']],
                                               'params': traindataset_info['params']}
        else:
            if 'params' not in traindataset_info:
                self.paras['train_dataset'] = {'backbone': getattr(import_module(traindataset_info['filepath']), traindataset_info['backbone']),
                                               'params': dict(datapath=self.config.train_dataset['datapath'])}
            else:
                self.paras['train_dataset'] = {'backbone': getattr(import_module(traindataset_info['filepath']), traindataset_info['backbone']),
                                               'params': traindataset_info['params']}
        if 'filepath' not in testdataset_info.keys() or not os.path.exists(testdataset_info['filepath']):
            if 'params' not in testdataset_info:
                self.paras['test_dataset'] = {'backbone': internal_env[self.config.test_dataset['backbone']],
                                              'params': dict(datapath=self.config.test_dataset['datapath'])}
            else:
                self.paras['test_dataset'] = {'backbone': internal_env[self.config.test_dataset['backbone']],
                                              'params': testdataset_info['params']}
        else:
            if 'params' not in testdataset_info:
                self.paras['test_dataset'] = {'backbone': getattr(import_module(testdataset_info['filepath']), testdataset_info['backbone']),
                                              'params': dict(datapath=self.config.test_dataset['datapath'])}
            else:
                self.paras['test_dataset'] = {'backbone': getattr(import_module(testdataset_info['filepath']), testdataset_info['backbone']),
                                              'params': testdataset_info['params']}
        # self.paras['train_dataset']={'backbone':internal_env[self.config.train_dataset['backbone']],'datapath':self.config.train_dataset['datapath']}
        # self.paras['test_dataset']={'backbone':internal_env[self.config.test_dataset['backbone']],'datapath':self.config.test_dataset['datapath']}
        '''import loss function anywhere'''
        if self.config.loss_fn is None:
            self.paras['loss_fn'] = {'backbone': MSELoss, 'params': dict()}
        else:
            loss_info = getattr(self.config, 'loss_fn')
            if 'params' not in loss_info.keys():
                loss_info['params'] = dict()
            if 'filepath' not in loss_info.keys() or not os.path.exists(loss_info['filepath']):

                self.paras['loss_fn'] = {'backbone': internal_env[self.config.loss_fn['backbone']],
                                         'params': loss_info['params']}
            else:
                self.paras['loss_fn'] = {'backbone': getattr(import_module(loss_info['filepath']), loss_info['backbone']),
                                         'params': loss_info['params']}
        if 'lr_config' in self.config_keys:
            self.paras['optimizer'] = dict(backbone=SGD,
                                           params=dict(lr=self.config.lr_config['init']))
            self.paras['schedular'] = dict(backbone=ReduceLROnPlateau,
                                           params=dict(patience=self.config.lr_config['patience']))
            warnings.warn(
                f"'lr_config' will be deprectaed in future versions, please specify 'optimizer' and 'schedular' in {self.configpath}, now optimizer has been set as SGD and schedular has been set as ReduceLROnPlateau")
        else:
            optimizer_info = getattr(self.config, 'optimizer')
            if 'params' not in optimizer_info.keys():
                optimizer_info['params'] = dict()
            if 'filepath' not in optimizer_info.keys() or not os.path.exists(optimizer_info['filepath']):
                self.paras['optimizer'] = dict(bakcbone=internal_env[optimizer_info['backbone']],
                                               params=optimizer_info['params'])
            else:
                self.paras['optimizer'] = dict(bakcbone=getattr(import_module(optimizer_info['filepath']), optimizer_info['backbone']),
                                               params=optimizer_info['params'])
            schedular_info = getattr(self.config, 'schedular')
            if 'params' not in schedular_info.keys():
                schedular_info['params'] = dict()
            if 'filepath' not in schedular_info.keys() or not os.path.exists(schedular_info['filepath']):
                self.paras['schedular'] = dict(backbone=internal_env[schedular_info['backbone']],
                                               params=schedular_info['params'])
            else:
                self.paras['schedular'] = dict(backbone=getattr(import_module(schedular_info['filepath']), schedular_info['backbone']),
                                               params=schedular_info['params'])
        self.paras['hyperpara'] = self.config.hyperpara
        self.paras['work_config'] = self.config.work_config
        self.paras['checkpoint_config'] = self.config.checkpoint_config
        self.paras['gpu_config'] = self.config.gpu_config
        self.paras['config'] = dict(path=self.configpath)

    def __repr__(self) -> str:
        info = ''
        for key in self.paras:
            info += f"{key}:\n\t{self.paras[key]}\n"
        return info
