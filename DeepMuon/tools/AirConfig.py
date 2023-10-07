'''
Author: airscker
Date: 2022-09-20 23:29:14
LastEditors: airscker
LastEditTime: 2023-10-04 16:09:46
Description: Import configuration file and prepare configurations for experiments

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''

import os
import time
import shutil
import warnings

import torch
import DeepMuon

from yapf.yapflib.yapf_api import FormatCode
from ..tools.AirFunc import import_module, readable_dict, module_source


class Config:
    """
    ## Load Training Configuration from Python File

    ### Args:
        - configpath: The path of the config file
        - config_module: If we don't want to import configuration from file, we can directly specify the module to be used

    ### Attributions:
        - paras: The parameters in config file, dtype: dict
            - mandatory:
                `model`,
                `train_dataset`,
                `test_dataset`,
                `work_config`,
                `checkpoint_config`,
                `loss_fn`,
                `hyperpara`,
                `optimizer`,
                `scheduler`
            - optional:
                `evaluation`,
                `fsdp_parallel`,
                `optimize_config`,
                `search_params`

    ### Example:
    
    >>> search_config=dict(search_space= {'seq_dropout': {'_type': 'choice', '_value': [0.1, 0.3, 0.5, 0.7]}},
                            exp_name='DeepMuon EXP',
                            concurrency=1,
                            trial_number=10,
                            port=14001,
                            tuner='TPE')
    >>> search_params = dict(seq_dropout=0.1)
    >>> model = dict(filepath='Mypath/model.py',
                    backbone='VST',
                    pipeline='regression',
                    params=dict(n_classes=11,
                                input_shape=(3, 130, 130),
                                seq_dropout=search_params['seq_dropout']))
    >>> train_dataset = dict(filepath='',
                            backbone='NIIDecodeV2',
                            collate_fn=None,
                            params=dict(ann_file=None,
                                        mask_ann=None,
                                        fusion=False,
                                        modalities=[],
                                        augment_pipeline=[dict(type='HistEqual'),
                                                        dict(type='SingleNorm'),
                                                        dict(type='Padding', size=(120, 120)),
                                                        dict(type='Resize', size=(130, 130))]))
    >>> test_dataset = dict(filepath='',
                            backbone='NIIDecodeV2',
                            collate_fn=None,
                            params=dict(ann_file=None,
                                        mask_ann=None,
                                        fusion=False,modalities=[],
                                        augment_pipeline=[dict(type='HistEqual'),
                                                        dict(type='SingleNorm'),
                                                        dict(type='Padding', size=(120, 120)),
                                                        dict(type='Resize', size=(130, 130))]))
    >>> work_config = dict(work_dir='./VST_1', logfile='log.log')
    >>> checkpoint_config = dict(load_from='', resume_from='', save_inter=50)
    >>> loss_fn = dict(filepath='',backbone='CrossEntropyLoss')
    >>> evaluation = dict(interval=1, metrics=['f1_score'])
    >>> optimizer = dict(filepath='',backbone='SGD', params=dict(lr=0.0001, momentum=0.9, nesterov=True))
    >>> scheduler = dict(filepath='',backbone='CosineAnnealingLR', params=dict(T_max=10))
    >>> hyperpara = dict(epochs=2000, batch_size=7500, inputshape=[1, 3, 40, 10, 10])
    >>> fsdp_parallel=dict(enabled=True,min_num_params=1e6)
    >>> optimize_config = dict(fp16=False, grad_acc=8, grad_clip=None, double_precision=False)
    """

    def __init__(self, configpath: str=None, config_module=None):
        self.paras = {'model': None,
                      'train_dataset': None,
                      'test_dataset': None,
                      'work_config': None,
                      'checkpoint_config': None,
                      'loss_fn': None,
                      'evaluation': None,
                      'hyperpara': None,
                      'optimizer': None,
                      'scheduler': None,
                      'lr_config': None, }
        if config_module is not None:
            self.config = config_module
        else:
            self.config = import_module(configpath)
        self.config_keys = dir(self.config)
        self.configpath = configpath
        self.__check_config()
        self.__para_config()
        # print(self.paras)

    def move_config(self,formatted=True,source_code:str='',save_path:str=None):
        if save_path is None:
            save_path=os.path.join(self.paras['work_config']['work_dir'], 'config.py')
        if self.configpath is not None:
            if formatted:
                source_code=FormatCode(module_source(self.configpath))[0]
                with open(save_path,'w+')as f:
                    f.write(source_code)
                f.close()
            else:
                shutil.copyfile(self.configpath,save_path)
        else:
            if formatted:
                source_code=FormatCode(source_code)[0]
            with open(save_path,'w+')as f:
                f.write(source_code)
            f.close()

    def __check_config(self):
        paras_config = self.config_keys
        error = []
        for key in self.paras.keys():
            if key not in paras_config:
                error.append(key)
        if 'lr_config' in error and 'optimizer' not in error and 'scheduler' not in error:
            error.remove('lr_config')
        if 'lr_config' not in error:
            if 'scheduler' in error:
                error.remove('scheduler')
            if 'optimizer' in error:
                error.remove('optimizer')
        if 'evaluation' in error:
            error.remove('evaluation')
        assert len(
            error) == 0, f'These basic configurations are not specified in {self.configpath}:\n{error}'
    def __load_dataset(self,dataset_info,dataset_module):
        backbone=getattr(dataset_module,dataset_info['backbone'])
        if 'collate_fn' not in dataset_info.keys() or dataset_info['collate_fn'] is None:
            collate_fn=None
        else:
            collate_fn=getattr(dataset_module,dataset_info['collate_fn'])
        if 'params' not in dataset_info:
            params={}
        else:
            params=dataset_info['params']
        if 'num_workers' not in params.keys():
            num_workers=0
        else:
            num_workers=params['num_workers']
        return backbone,collate_fn,params,num_workers

    def __para_config(self):
        internal_env = globals()

        '''import the model anywhere'''
        model_info = getattr(self.config, 'model')
        if 'pipeline' not in model_info.keys():
            pipeline='regression'
        else:
            pipeline=model_info['pipeline']
        if 'params' not in model_info.keys():
            model_params = {}
        else:
            model_params = model_info['params']
        if 'filepath' not in model_info.keys() or not os.path.exists(model_info['filepath']):
            self.paras['model'] = {'backbone': getattr(DeepMuon.models, model_info['backbone']),
                                   'pipeline':getattr(DeepMuon.train.pipeline,pipeline),
                                   'params': model_params}
        else:
            imported_module=import_module(model_info['filepath'])
            self.paras['model'] = {'backbone': getattr(imported_module,model_info['backbone']),
                                   'pipeline':getattr(imported_module,pipeline),
                                   'params': model_params}
        # self.paras['model']={'backbone':internal_env[self.config.model['backbone']]}
        
        '''import dataset anywhere'''
        traindataset_info = getattr(self.config, 'train_dataset')
        testdataset_info = getattr(self.config, 'test_dataset')
        if 'filepath' not in traindataset_info.keys() or not os.path.exists(traindataset_info['filepath']):
            # backbone=internal_env[self.config.train_dataset['backbone']]
            dataset_module=DeepMuon.dataset
            backbone,collate_fn,params,num_workers=self.__load_dataset(traindataset_info,dataset_module)
        else:
            imported_module=import_module(traindataset_info['filepath'])
            backbone,collate_fn,params,num_workers=self.__load_dataset(traindataset_info,imported_module)
        self.paras['train_dataset']={'backbone':backbone,'collate_fn':collate_fn,'num_workers':num_workers,'params':params}
        if 'filepath' not in testdataset_info.keys() or not os.path.exists(testdataset_info['filepath']):
            dataset_module=DeepMuon.dataset
            backbone,collate_fn,params,num_workers=self.__load_dataset(testdataset_info,dataset_module)
        else:
            imported_module=import_module(testdataset_info['filepath'])
            backbone,collate_fn,params,num_workers=self.__load_dataset(testdataset_info,imported_module)
        self.paras['test_dataset']={'backbone':backbone,'collate_fn':collate_fn,'num_workers':num_workers,'params':params}
        # self.paras['train_dataset']={'backbone':internal_env[self.config.train_dataset['backbone']],'datapath':self.config.train_dataset['datapath']}
        # self.paras['test_dataset']={'backbone':internal_env[self.config.test_dataset['backbone']],'datapath':self.config.test_dataset['datapath']}

        '''import loss function anywhere'''
        if self.config.loss_fn is None:
            self.paras['loss_fn'] = {'backbone': torch.nn.MSELoss, 'params': dict()}
        else:
            loss_info = getattr(self.config, 'loss_fn')
            if 'params' not in loss_info.keys():
                loss_info['params'] = dict()
            if 'filepath' not in loss_info.keys() or not os.path.exists(loss_info['filepath']):
                # self.paras['loss_fn'] = {'backbone': internal_env[self.config.loss_fn['backbone']],
                #                          'params': loss_info['params']}
                try:
                    self.paras['loss_fn'] = {'backbone': getattr(DeepMuon.loss_fn, loss_info['backbone']),
                                             'params': loss_info['params']}
                except:
                    self.paras['loss_fn'] = {'backbone': getattr(torch.nn.modules.loss, loss_info['backbone']),
                                            'params': loss_info['params']}
            else:
                self.paras['loss_fn'] = {'backbone': getattr(import_module(loss_info['filepath']), loss_info['backbone']),
                                         'params': loss_info['params']}
        if 'lr_config' in self.config_keys:
            self.paras['optimizer'] = dict(backbone=torch.optim.SGD,
                                           params=dict(lr=self.config.lr_config['init']))
            self.paras['scheduler'] = dict(backbone=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                           params=dict(patience=self.config.lr_config['patience']))
            warnings.warn(
                f"'lr_config' will be deprecated in future versions, please specify 'optimizer' and 'scheduler' in {self.configpath}, now optimizer has been set as SGD and scheduler has been set as ReduceLROnPlateau")
        else:
            optimizer_info = getattr(self.config, 'optimizer')
            if 'params' not in optimizer_info.keys():
                optimizer_info['params'] = dict()
            if 'filepath' not in optimizer_info.keys() or not os.path.exists(optimizer_info['filepath']):
                self.paras['optimizer'] = dict(backbone=getattr(torch.optim, optimizer_info['backbone']),
                                               params=optimizer_info['params'])
            else:
                self.paras['optimizer'] = dict(backbone=getattr(import_module(optimizer_info['filepath']), optimizer_info['backbone']),
                                               params=optimizer_info['params'])
            scheduler_info = getattr(self.config, 'scheduler')
            if 'params' not in scheduler_info.keys():
                scheduler_info['params'] = dict()
            if 'filepath' not in scheduler_info.keys() or not os.path.exists(scheduler_info['filepath']):
                self.paras['scheduler'] = dict(backbone=getattr(torch.optim.lr_scheduler, scheduler_info['backbone']),
                                               params=scheduler_info['params'])
            else:
                self.paras['scheduler'] = dict(backbone=getattr(import_module(scheduler_info['filepath']), scheduler_info['backbone']),
                                               params=scheduler_info['params'])

        '''import other hyperparameters configuration'''
        self.paras['hyperpara'] = self.config.hyperpara
        self.paras['work_config'] = self.config.work_config
        if 'logfile' not in self.paras['work_config'].keys():
            current_time = time.strftime(
                "%Y-%m-%d-%H.%M.%S", time.localtime(time.time()))
            self.paras['work_config']['logfile'] = f"{current_time}.log"
        self.paras['checkpoint_config'] = self.config.checkpoint_config
        self.paras['config'] = dict(path=self.configpath)
        if 'optimize_config' not in self.config_keys:
            self.paras['optimize_config'] = dict(
                fp16=False, grad_acc=1, grad_clip=None,double_precision=False,find_unused_parameters=False)
        else:
            optim_config = getattr(self.config, 'optimize_config')
            if 'fp16' not in optim_config.keys():
                optim_config['fp16'] = False
            if 'grad_acc' not in optim_config.keys():
                optim_config['grad_acc'] = 1
            if 'grad_clip' not in optim_config.keys():
                optim_config['grad_clip'] = None
            if 'double_precision' not in optim_config.keys():
                optim_config['double_precision'] = False
            if optim_config['double_precision']:
                self.precision=torch.DoubleTensor
            else:
                self.precision=torch.FloatTensor
            if 'find_unused_parameters' not in optim_config.keys():
                optim_config['find_unused_parameters']=False
            self.paras['optimize_config'] = optim_config
        if 'gpu_config' in self.config_keys:
            warnings.warn(
                "'gpu_config' was deprecated, please do not use it anymore")
        if 'fsdp_parallel' in self.config_keys:
            fsdp_op = getattr(self.config, 'fsdp_parallel')
            if 'enabled' not in fsdp_op.keys():
                fsdp_op['enabled'] = False
                fsdp_op['min_num_params'] = 0
                warnings.warn(
                    f"keyword 'enabled' not epcified in fsdp_parallel, set fsdp_parallel.enabled as False and fsdp_parallel.min_num_params as 0")
            if 'min_num_params' not in fsdp_op.keys():
                fsdp_op['enabled'] = False
                fsdp_op['min_num_params'] = 0
                warnings.warn(
                    f"keyword 'min_num_params' not epcified in fsdp_parallel, set fsdp_parallel.enabled as False and fsdp_parallel.min_num_params as 0")
            self.paras['fsdp_parallel'] = fsdp_op
        else:
            self.paras['fsdp_parallel'] = dict(enabled=False, min_num_params=0)
        if 'evaluation' in self.config_keys:
            evaluation_op = getattr(self.config, 'evaluation')
            if 'interval' not in evaluation_op.keys():
                warnings.warn(
                    f"'interval' in evaluation command expected, however {evaluation_op} given, interval is set as 1 defaultly to avoid errors")
                evaluation_op['interval'] = 1
            if 'metrics' not in evaluation_op.keys():
                warnings.warn(
                    f"'metrics' in evaluation command expected, however {evaluation_op} given, metrics is set as empty defaultly to avoid errors")
                eva_metrics = {}
            else:
                eva_metrics = {}
                for ops in evaluation_op['metrics']:
                    # if ops not in internal_env.keys():
                    if ops not in DeepMuon.loss_fn.__dir__():
                        warnings.warn(
                            f"evaluaction metrics '{ops}' doesn't exists!")
                        continue
                    # eva_metrics[ops] = internal_env[ops]
                    eva_metrics[ops]=getattr(DeepMuon.loss_fn,ops)
            if 'sota_target' not in evaluation_op.keys():
                evaluation_op['sota_target'] = dict(mode='min', target=None)
            else:
                assert evaluation_op['sota_target'][
                    'mode'] == 'min' or 'max', f"mode='min'/'max' expected in evaluation command, however {evaluation_op} given"
                assert evaluation_op['sota_target']['target'] == None or evaluation_op['sota_target']['target'] in evaluation_op[
                    'metrics'], f"'target' in evaluation command should be None/within metrics, however {evaluation_op} given"
                # evaluation_op['sota_target']['target'] = internal_env[evaluation_op['sota_target']['target']]
            self.paras['evaluation'] = dict(
                interval=evaluation_op['interval'], metrics=eva_metrics, sota_target=evaluation_op['sota_target'])
        else:
            self.paras['evaluation'] = dict(
                interval=1, metrics={}, sota_target=dict(mode='min', target=None))

    def __repr__(self) -> str:
        return readable_dict(self.paras)
