'''
Author: airscker
Date: 2023-09-29 00:41:08
LastEditors: airscker
LastEditTime: 2023-09-29 01:28:11
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
from threading import Thread
from queue import Queue
from typing import Callable,Iterable,Any

class TaskFIFOQueue(Thread):
    '''
    ## Multithread task FIFO quene

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
        >>>     tasks.add_task(save_data,(data,i),i)
        >>> tasks.end_task()

        Output:
        >>> Task 0 added
        >>> Task 1 added
        >>> Task 2 added
        >>> Task 3 added
        >>> Task 4 added
        >>> Task 0 done
        >>> Task 4 done
    '''
    def __init__(self,sequenced=True,verbose=False,daemon=True,**kwargs):
        super().__init__(daemon=daemon,**kwargs)
        self.sequenced = sequenced
        self._quene=Queue()
        self.verbose=verbose
    def add_task(self,task:Callable,args:Iterable=(),task_id:Any=None):
        '''
        ## Add task to quene

        ### Args:
            - task: [Callable] task to be added, such as `torch.save`.
            - args: [Iterable] arguments of task, such as `(data_0,data_1)`.
            - task_id: [Any] task id, such as `0`, can be omited.
        '''
        self._quene.put(((task,*args),task_id))
        if self.verbose:
            print(f"Task {task_id} added")
    def run(self):
        while True:
            task,task_id = self._quene.get()
            if not self.sequenced:
                if self._quene.qsize()>0:
                    self._quene.task_done()
                    continue
            task[0](*task[1:])
            if self.verbose:
                print(f"Task {task_id} done")
            self._quene.task_done()
    def end_task(self):
        self._quene.join()