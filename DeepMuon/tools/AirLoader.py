'''
Author: airscker
Date: 2023-07-25 18:35:03
LastEditors: airscker
LastEditTime: 2023-09-05 18:31:42
Description: Multi-thread data loading base tools.

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import threading
import warnings
import math
import os
from typing import Any,Callable

class MultiThreadLoaderBase(threading.Thread):
    def __init__(self,LoadList:list[str],LoadMethod:Callable):
        super().__init__()
        self.loadlist=LoadList
        self.loadmethod=LoadMethod
        self.data=[]
    def run(self):
        for i in range(len(self.loadlist)):
            self.data.append(self.loadmethod(self.loadlist[i]))

def MultiThreadLoader(LoadList:list[Any],ThreadNum:int,LoadMethod:Callable):
    '''
    ## Multi thread data loading tool.

    ### Args:
        - LoadList: The path list of data to be loaded.
        - ThreadNum: The number of threads for multithread data loading.
        - LoadMethod: The method to load data.

    ### Returns:
        - all_data: The data list loaded, every element corresponds to the data loaded by the corresponding path in `LoadList`.
    '''
    if ThreadNum<1:
        warnings.warn(f'The number of threads for multithread data loading must be larger than 0, however, {ThreadNum} was given, we will reset it as {min(len(LoadList),5)}')
        ThreadNum=min(len(LoadList),5)
    if ThreadNum>len(LoadList):
        warnings.warn(f'The number of threads for multithread data loading must be smaller than the size of dataset, however, {ThreadNum} was given, we will reset it as {min(len(LoadList),5)}')
        ThreadNum=min(len(LoadList),5)
    core_num=os.cpu_count()
    if ThreadNum>core_num:
        warnings.warn(f'The number of cores of you CPU is {core_num}, however, {ThreadNum} threads was specified, we recomend you to set `ThreadNum` as {core_num} instead.')
    all_data=[]
    threads=[]
    subgroups=[]
    subgroup_len=math.ceil(len(LoadList)/ThreadNum)
    for i in range(ThreadNum):
        subgroups.append(LoadList[i*subgroup_len:(i+1)*subgroup_len])
    for i in range(len(subgroups)):
        threads.append(MultiThreadLoaderBase(LoadList=subgroups[i],LoadMethod=LoadMethod))
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        all_data+=threads[i].data
    return all_data