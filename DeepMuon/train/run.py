'''
Author: airscker
Date: 2022-10-07 21:35:54
LastEditors: airscker
LastEditTime: 2023-10-08 02:25:41
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''

import os
import argparse
import DeepMuon
import warnings
import platform
import subprocess
from multiprocessing import Process
from functools import partial
from ..tools.AirFunc import import_module,check_port
from ..tools.AirServer.FileSaving import FileSavingServer
OS=platform.system().lower()

pkg_path = DeepMuon.__path__[0]
try:
    from nni.experiment import Experiment
    NNHS_enabled=True
except:
    NNHS_enabled=False

class NNHSearch:
    '''
    The NNHSearch class is designed to facilitate neural network hyperparameter searching (NNHS) using the NNI (Neural Network Intelligence) package.
    It initializes the NNI experiment and sets up the configuration for the search space, experiment name, concurrency, trial number, port, tuner, and optimization mode.
    It also provides a method to start the experiment and another method to fix the port number to avoid port occupation errors in multi-concurrency searching.
    '''
    def __init__(self,path,search) -> None:
        self.config_info=import_module(path)
        self.search=search
        self.__init_NNS()
    def report_error(self,para_name):
        warnings.warn(f'Element `{para_name}` not found in configuartion file, neural network hyperparameter searching will be disabled.')
        self.search=False
    def __init_NNS(self):
        global NNHS_enabled
        if not NNHS_enabled:
            warnings.warn('NNI unavailable on this platform, neural network hyperparameter searching will be disabled')
            self.search=False
            return 0
        if self.search:
            elements=dir(self.config_info)
            if 'search_config' not in elements:
                self.report_error('search_config')
            if 'search_params' not in elements:
                self.report_error('search_params')
            search_config=self.config_info.search_config
            if 'search_space' not in search_config.keys():
                self.report_error('search_space')
            else:
                search_space=search_config['search_space']
            if 'exp_name' not in search_config.keys():
                exp_name='DeepMuon EXP'
            else:
                exp_name=str(search_config['exp_name'])
            if 'concurrency' not in search_config.keys():
                concurrency=1
            else:
                concurrency=search_config['concurrency']
            if 'trail_number' not in search_config.keys():
                trail_number=10
            else:
                trail_number=search_config['trail_number']
            if 'port' not in search_config.keys():
                port=23328
            else:
                port=search_config['port']
            if 'tuner' not in search_config.keys():
                tuner='TPE'
            else:
                tuner=search_config['tuner']
            if 'evaluation' in elements:
                if 'sota_target' in self.config_info.evaluation.keys():
                    sota_target=self.config_info.evaluation['sota_target']
                    if 'mode' in sota_target.keys():
                        mode=sota_target['mode']
                        if mode=='max':
                            optimize_mode='maximize'
                        elif mode=='min':
                            optimize_mode='minimize'
                        else:
                            optimize_mode='minimize'
            else:
                optimize_mode='minimize'
            work_dir=self.config_info.work_config['work_dir']
            self.experiment=Experiment('local')
            self.experiment.config.trial_concurrency=concurrency
            self.experiment.config.experiment_name=exp_name
            self.experiment.config.trial_code_directory=os.path.join(pkg_path,'train')
            self.experiment.config.search_space=search_space
            self.experiment.config.tuner.name=tuner
            self.experiment.config.tuner.class_args['optimize_mode']=optimize_mode
            self.experiment.config.max_trial_number=trail_number
            self.experiment.config.experiment_working_directory=os.path.join(work_dir,'NNHS_log')
            self.NNHS_port=port
    def fix_port(self,port):
        new_port=port
        while True:
            port_usable=check_port(port=new_port)
            if not port_usable:
                new_port+=1
            else:
                break
        if new_port!=port:
            warnings.warn(f'Port {port} is unavailable, we reset it as the nearest usable port {new_port}')
        return new_port
    def start_exp(self,exp_args):
        '''Set CUDA Environment'''
        env = ''
        for i in range(len(exp_args.gpus)):
            env += f'{exp_args.gpus[i]}'
            if i+1 < len(exp_args.gpus):
                env += ','
        # '''Set Training File'''
        # if not exp_args.train.endswith('.py'):
        #     exp_args.train += '.py'
        file = os.path.join(pkg_path, 'train/dist_train.py')
        '''Find Usable Port, Avoid ERROR in Multi-Concurrency Searching'''
        base_command=f'CUDA_VISIBLE_DEVICES={env} ' if OS =='linux' else ''
        if self.search and self.experiment.config.trial_concurrency>1:
            print(f"NNHS concurrency is {self.experiment.config.trial_concurrency}, which is bigger than 1, to avoid errors brought by port occupation, we set PORTs of multi concurrency NNHS experiments randomly.")
            base_command = base_command + f'torchrun --nproc_per_node={len(exp_args.gpus)} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 {file} --config {exp_args.config}'
        else:
            exp_args.port=self.fix_port(int(exp_args.port))
            base_command = base_command + f'torchrun --nproc_per_node={len(exp_args.gpus)} --master_port {exp_args.port} {file} --config {exp_args.config}'
        if exp_args.test!='':
            if self.search:
                warnings.warn('Neural network hyperparameter searching is unavailable in testing mode.')
                self.search=False
            base_command=base_command+f' --test {exp_args.test}'
            os.system(base_command)
            return 0
        else:
            if self.search:
                self.NNHS_port=self.fix_port(self.NNHS_port)
                base_command=base_command+' --search'
                self.experiment.config.trial_command=base_command
                self.experiment.run(self.NNHS_port)
                self.experiment.stop()
                return 0
            else:
                if exp_args.server:
                    base_command=base_command+' --server'
                    run_model=Process(target=os.system,args=(base_command,))
                    run_server=Process(target=partial(FileSavingServer,verbose=False,workdir=self.config_info.work_config['work_dir']),args=())
                    run_server.start()
                    run_model.start()
                    run_model.join()
                    run_server.join()
                else:
                    os.system(base_command)
                return 0


def main():
    global NNHS_enabled
    global pkg_path
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', nargs='+', default=0, type=int,help='GPU ID(s) to use, e.g. 0 1 2 3')
    parser.add_argument('-p', '--port', default=22911, type=int,help='Port number for multi-process training, e.g. 22911')
    parser.add_argument('-c', '--config', default='', type=str,help='Configuration file path, e.g. ./config.py')
    parser.add_argument('-ts', '--test', default='', type=str,help='If the path of tested checkpoint given, then test mode will be activated, e.g. ./checkpoint.pth')
    parser.add_argument('-sr', '--search', action='store_true',help='If this flag is set, then neural network hyperparameter searching will be activated.')
    parser.add_argument('-sv', '--server', action='store_true',help='If this flag is set, then the file saving server will be started.')
    args = parser.parse_args()
    '''Set NNHS Configuration'''
    exp: NNHSearch = NNHSearch(args.config, args.search)
    '''Start Experiment'''
    exp.start_exp(args)
    # print(command)


if __name__ == '__main__':
    main()
