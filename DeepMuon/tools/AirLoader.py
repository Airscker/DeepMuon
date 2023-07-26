'''
Author: airscker
Date: 2023-07-25 18:35:03
LastEditors: airscker
LastEditTime: 2023-07-26 09:28:22
Description: Multi-thread data loading base tools.

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import threading
import warnings
import math
from typing import Any

class MultiThreadLoaderBase(threading.Thread):
    def __init__(self,LoadList:list[str],LoadMethod:Any):
        super().__init__()
        self.loadlist=LoadList
        self.loadmethod=LoadMethod
        self.data=[]
    def run(self):
        for i in range(len(self.loadlist)):
            self.data.append(self.loadmethod(self.loadlist[i]))

def MultiThreadLoader(LoadList:list[Any],ThreadNum:int,LoadMethod:Any):
    if ThreadNum<1:
        warnings.warn(f'The number of threads for multithread data loading must be larger than 0, however, {ThreadNum} was given, we will reset it as {min(len(LoadList),5)}')
        ThreadNum=min(len(LoadList),5)
    if ThreadNum>len(LoadList):
        warnings.warn(f'The number of threads for multithread data loading must be smaller than the size of dataset, however, {ThreadNum} was given, we will reset it as {min(len(LoadList),5)}')
        ThreadNum=min(len(LoadList),5)
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
        