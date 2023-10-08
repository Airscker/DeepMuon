'''
Author: airscker
Date: 2023-09-29 00:41:08
LastEditors: airscker
LastEditTime: 2023-10-08 01:50:56
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import queue
import socket
import pickle as pkl
import multiprocessing
from threading import Thread
from typing import Callable,Iterable,Any
from collections.abc import Callable, Iterable, Mapping

class TaskFIFOQueueThread(Thread):
    '''
    ## Multithread task FIFO quene based on multi-threading

    ### Args:
        - sequenced: [bool] if True, the task will be executed sequencially, otherwise, only the newest task will be excuted.
        - verbose: [bool] if True, the task will print the task id when added and done.
        - daemon: [bool] if True, the thread will be set as daemon.
        - kwargs: [dict] other arguments of `Thread`.

    ### Methods:
        - add_task: add tasks to the quene.
        - end_task: wait until all tasks are done, MUST be called at the program ends.
    
    ### Example:
        Code:
        >>> import torch
        >>> from Deepmuon.tools import TaskFIFOQueueThread
        >>> data=torch.rand(1000,1000)
        >>> def save_data(data,index):
                torch.save(data, f'./test/data_{index}.pt')
                time.sleep(3)
        >>> tasks=TaskFIFOQueueThread(sequenced=False,verbose=True,daemon=True)
        >>> tasks.start()
        >>> for i in range(100):
        >>>     tasks.add_task(save_data,dict(data=data,index=i),i)
        >>> tasks.end_task()

        Output:
        >>> Task 0 added
        >>> Task 1 added
        >>> Task 2 added
        >>> Task 3 added
        >>> Task 4 added
        >>> Task 0 done
        >>> Task 4 done
        >>> Task terminated
    '''
    def __init__(self,sequenced=True,verbose=False,daemon=True,**kwargs):
        super().__init__(daemon=daemon,**kwargs)
        self.sequenced = sequenced
        self._quene=queue.Queue()
        self.verbose=verbose
    def add_task(self,task:Callable,kwargs:dict={},task_id:Any=None):
        '''
        ## Add task to quene

        ### Args:
            - task: [Callable] task to be added, such as `torch.save`.
            - kwargs: [dict] arguments of task, such as `dict(a=data_0,b=data_1)`.
            - task_id: [Any] task id, such as `0`, can be omited.
        '''
        self._quene.put(((task,kwargs),task_id))
        if self.verbose:
            print(f"Task {task_id} added")
        
    def run(self):
        while True:
            task,task_id = self._quene.get()
            if not self.sequenced:
                if self._quene.qsize()>0:
                    self._quene.task_done()
                    continue
            task[0](**task[1])
            if self.verbose:
                print(f"Task {task_id} done")
            self._quene.task_done()
    def end_task(self):
        self._quene.join()
        if self.verbose:
            print('Task terminated')

class TaskFIFOQueueProcess(multiprocessing.Process):
    '''
    ## Multithread task FIFO quene based on multiprocessing

    ### Args:
        - sequenced: [bool] if True, the task will be executed sequencially, otherwise, only the newest task will be excuted.
        - verbose: [bool] if True, the task will print the task id when added and done.
        - daemon: [bool] if True, the thread will be set as daemon.
        - kwargs: [dict] other arguments of `Thread`.

    ### Methods:
        - add_task: add tasks to the quene.
        - end_task: wait until all tasks are done, MUST be called at the program ends.
    
    ### Example:
        Code:
        >>> import torch
        >>> from Deepmuon.tools import TaskFIFOQuene
        >>> data=torch.rand(1000,1000)
        >>> def save_data(data,index):
                torch.save(data, f'./test/data_{index}.pt')
                time.sleep(3)
        >>> tasks=TaskFIFOQuene(sequenced=False,verbose=True,daemon=True)
        >>> tasks.start()
        >>> for i in range(100):
        >>>     tasks.add_task(save_data,dict(data=data,index=i),i)
        >>> tasks.end_task()

        Output:
        >>> Task 0 added
        >>> Task 1 added
        >>> Task 2 added
        >>> Task 3 added
        >>> Task 4 added
        >>> Task 0 skipped
        >>> Task 1 skipped
        >>> Task 2 skipped
        >>> Task 3 skipped
        >>> Task 4 done
        >>> Task terminated
    '''
    def __init__(self,sequenced=True,verbose=False,daemon=True,**kwargs):
        super().__init__(daemon=daemon,**kwargs)
        self.sequenced = sequenced
        self._quene=multiprocessing.Queue()
        self.verbose=verbose
        self.last_task_group=None
        self.last_done_task=None
    def add_task(self,task:Callable,kwargs:dict={},task_id:Any=None):
        '''
        ## Add task to quene

        ### Args:
            - task: [Callable] task to be added, such as `torch.save`.
            - kwargs: [dict] arguments of task, such as `dict(a=data_0,b=data_1)`.
            - task_id: [Any] task id, such as `0`, can be omited.
        '''
        self._quene.put(((task,kwargs),task_id,False))
        self.last_task_group=(task,kwargs)
        if self.verbose:
            print(f"Task {task_id} added")
    def run_task(self,task_group):
        task_group[0](**task_group[1])
        '''Clean the last task group to help verify whether at least the last task is done'''
        self.last_done_task=task_group
    def run(self):
        while True:
            task,task_id,stop = self._quene.get()
            if stop:
                '''If the last task is skipped accidentally, do it'''
                if self.last_done_task is None:
                    if self.verbose:
                        print('Last task is skipped, we fixed it')
                    task[0](**task[1])
                if self.verbose:
                    print('Task terminated')
                break
            if not self.sequenced:
                if self._quene.qsize()>0 and not stop:
                    if self.verbose:
                        print(f'Task {task_id} skipped')
                    continue
            self.run_task(task)
            if self.verbose:
                print(f"Task {task_id} done")
    def end_task(self):
        self._quene.put((self.last_task_group,'TERMINATE',True))
        self.join()
        self.terminate()