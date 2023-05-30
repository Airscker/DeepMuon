'''
Author: airscker
Date: 2023-05-30 21:21:31
LastEditors: airscker
LastEditTime: 2023-05-30 22:54:57
Description: NULL

Copyright (C) OpenMMLab. All rights reserved.
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import sys
import GPUtil
import subprocess
from prettytable import PrettyTable

import torch

class EnvINFO:
    def __init__(self) -> None:
        self.env_info={}
        self.__init()
        
    def __get_build_config(self):
        if self.torch_version == 'parrots':
            from parrots.config import get_build_info
            self.env_info['PyTorch compiling details']=get_build_info()
        else:
            self.env_info['PyTorch compiling details']=torch.__config__.show().rstrip('\n')
    def __is_rocm_pytorch(self)->bool:
        is_rocm = False
        if self.torch_version != 'parrots':
            try:
                from torch.utils.cpp_extension import ROCM_HOME
                is_rocm = True if ((torch.version.hip is not None) and
                                (ROCM_HOME is not None)) else False
            except ImportError:
                pass
        return is_rocm
    def __get_cuda_home(self):
        if self.torch_version == 'parrots':
            from parrots.utils.build_extension import CUDA_HOME
        else:
            if self.__is_rocm_pytorch():
                from torch.utils.cpp_extension import ROCM_HOME
                CUDA_HOME = ROCM_HOME
            else:
                from torch.utils.cpp_extension import CUDA_HOME
        self.env_info['CUDA_HOME']=CUDA_HOME
        self.cuda_home=CUDA_HOME
    def __get_nvcc_version(self):
        if self.cuda_home is not None and os.path.isdir(self.cuda_home):
            try:
                nvcc = os.path.join(self.cuda_home, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
                release = nvcc.rfind('Cuda compilation tools')
                build = nvcc.rfind('Build ')
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
        self.env_info['NVCC']=nvcc
    def __get_gcc_version(self):
        try:
            # Check C++ Compiler.
            # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
            # indicating the compiler used, we use this to get the compiler name
            import sysconfig
            cc = sysconfig.get_config_var('CC')
            if cc:
                cc = os.path.basename(cc.split()[0])
                cc_info = subprocess.check_output(f'{cc} --version', shell=True)
                self.env_info['GCC'] = cc_info.decode('utf-8').partition(
                    '\n')[0].strip()
            else:
                # on Windows, cl.exe is not in PATH. We need to find the path.
                # distutils.ccompiler.new_compiler() returns a msvccompiler
                # object and after initialization, path to cl.exe is found.
                import locale
                from distutils.ccompiler import new_compiler
                ccompiler = new_compiler()
                ccompiler.initialize()
                cc = subprocess.check_output(
                    f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
                try:
                    encoding = os.device_encoding(
                        sys.stdout.fileno()) or locale.getpreferredencoding()
                    self.env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
                except:
                    pass
                self.env_info['GCC'] = 'n/a'
        except subprocess.CalledProcessError:
            self.env_info['GCC'] = 'n/a'
    def __get_gpu_info(self):
        GPU_info=PrettyTable(['ID','MEMORY TOTAL (MB)','NAME','DRIVER VERSION'])
        GPU_group=GPUtil.getGPUs()
        for id,gpu in enumerate(GPU_group):
            GPU_info.add_row([gpu.id,gpu.memoryTotal,gpu.name,gpu.driver])
        self.env_info['GPU devices']=f'\n{GPU_info}\n'
    def __init(self)->dict:
        """
        ## Collect the information of the running environments.

        About the environment information. The following fields are contained:
            - Platform: The variable of `sys.platform`.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - CUDA_HOME (optional): The env var `CUDA_HOME`.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of ``torch.__config__.show()``.
            - GPU devices: Detailed information of every GPU.
        """
        self.platform=sys.platform
        self.python_version=sys.version.replace('\n','')
        self.torch_version=torch.__version__
        self.cuda_available=torch.cuda.is_available()
        self.env_info['Platform']=self.platform
        self.env_info['Python']=self.python_version
        self.env_info['CUDA available']=self.cuda_available
        self.__get_cuda_home()
        self.__get_nvcc_version()
        self.__get_gcc_version()
        self.env_info['PyTorch']=self.torch_version
        self.__get_build_config()
        self.__get_gpu_info()
        return self.env_info
    def __repr__(self)->str:
        return 'Environment Information:\n'+'\n'.join([(f'{k}: {v}') for k, v in self.env_info.items()])