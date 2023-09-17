'''
Author: airscker
Date: 2023-07-25 18:35:03
LastEditors: airscker
LastEditTime: 2023-09-16 12:50:17
Description: Multi-thread data loading base tools.

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import threading
import math
import os
from typing import Any,Callable
from tqdm import tqdm

class MultiThreadLoaderBase(threading.Thread):
    def __init__(self,LoadList:list[str],LoadMethod:Callable,verbose:bool=False):
        super().__init__()
        self.loadlist=LoadList
        self.loadmethod=LoadMethod
        self.verbose=verbose
        self.data=[]
    def run(self):
        if self.verbose:
            bar=tqdm(range(len(self.loadlist)),desc=f'Thread {threading.get_ident()}:',mininterval=1)
        else:
            bar=range(len(self.loadlist))
        for i in bar:
            self.data.append(self.loadmethod(self.loadlist[i]))

def MultiThreadLoader(LoadList:list[Any],ThreadNum:int,LoadMethod:Callable,verbose:bool=False):
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
        print(f'The number of threads for multithread data loading must be larger than 0, however, {ThreadNum} was given, we will reset it as {len(LoadList)}')
        ThreadNum=len(LoadList)
    if ThreadNum>len(LoadList):
        print(f'The number of threads for multithread data loading must be smaller than the size of dataset, however, {ThreadNum} was given, we will reset it as {len(LoadList)}')
        ThreadNum=len(LoadList)
    core_num=os.cpu_count()
    if ThreadNum>core_num:
        print(f'The number of cores of you CPU is {core_num}, however, {ThreadNum} threads was specified, we recomend you to set `ThreadNum` as {core_num} instead.')
    all_data=[]
    threads=[]
    subgroups=[]
    subgroup_len=math.ceil(len(LoadList)/ThreadNum)
    for i in range(ThreadNum):
        subgroups.append(LoadList[i*subgroup_len:(i+1)*subgroup_len])
    for i in range(len(subgroups)):
        threads.append(MultiThreadLoaderBase(LoadList=subgroups[i],LoadMethod=LoadMethod,verbose=verbose))
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        all_data+=threads[i].data
    return all_data