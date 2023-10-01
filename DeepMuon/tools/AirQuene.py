'''
Author: airscker
Date: 2023-09-29 00:41:08
LastEditors: airscker
LastEditTime: 2023-09-30 01:10:43
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
        >>> Task terminated
    '''
    def __init__(self,sequenced=True,verbose=False,daemon=True,**kwargs):
        super().__init__(daemon=daemon,**kwargs)
        self.sequenced = sequenced
        self._quene=queue.Queue()
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
        >>>     tasks.add_task(save_data,(data,i),i)
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
    def add_task(self,task:Callable,args:Iterable=(),task_id:Any=None):
        '''
        ## Add task to quene

        ### Args:
            - task: [Callable] task to be added, such as `torch.save`.
            - args: [Iterable] arguments of task, such as `(data_0,data_1)`.
            - task_id: [Any] task id, such as `0`, can be omited.
        '''
        self._quene.put(((task,*args),task_id,False))
        self.last_task_group=(task,*args)
        if self.verbose:
            print(f"Task {task_id} added")
    def run_task(self,task_group):
        task_group[0](*task_group[1:])
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
                    task[0](*task[1:])
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


class Client(object):
    def __init__(self, serverIP:str='127.0.0.1', serverPort:int=11000, verbose:bool=False):
        self.serverIP = serverIP
        self.serverPort = serverPort
        self.verbose = verbose
        self.base_Buffsize = 1024
        self.ADDR = (self.serverIP, self.serverPort)
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect(self.ADDR)
        if self.verbose:
            print(f'Connected to server {self.ADDR}')

    def send(self, data):
        data_bytes=pkl.dumps(data)
        self.clientSocket.send(pkl.dumps(len(data_bytes)))
        self.clientSocket.send(data_bytes)
        if data == 'ENDPORT':
            return 0
        response=self.receive()
        if self.verbose:
            print(f'Sent data to server {self.ADDR} with {len(data_bytes)} bytes.')
        if response == 'SUCCESS':
            if self.verbose:
                print('Server receivement DONE.')
            return 1
        else:
            if self.verbose:
                print('Server receivement FIALED.')
            return 0
    def receive(self):
        databytes_len=self.clientSocket.recv(self.base_Buffsize)
        data = self.clientSocket.recv(pkl.loads(databytes_len))
        if self.verbose:
            print(f'Received data from server {self.ADDR} with {len(data)} bytes.')
        return pkl.loads(data)

    def close(self):
        self.send('ENDPORT')
        self.clientSocket.close()


class Server(object):
    def __init__(self, serverIP:str='127.0.0.1', serverPort:int=11000, listenNum:int=1, verbose:bool=False):
        self.serverIP = serverIP
        self.serverPort = serverPort
        self.verbose = verbose
        self.base_Buffsize = 1024
        self.ADDR = (self.serverIP, self.serverPort)
        self.tcpSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpSocket.bind(self.ADDR)
        self.tcpSocket.listen(listenNum)
        self.data_queue = []
        if self.verbose:
            print(f'Prepared to receive data from {self.ADDR}')
        end_signal=self.receive()
        if end_signal == 'ENDPORT':
            self.close()
    def send(self, data):
        data_bytes=pkl.dumps(data)
        self.clientSocket.send(pkl.dumps(len(data_bytes)))
        self.clientSocket.send(data_bytes)
    def receive(self):
        while True:
            self.clientSocket, self.clientAddr = self.tcpSocket.accept()
            if self.verbose:
                print(f'Connected from client: {self.clientAddr}')
            while True:
                try:
                    databytes_len = self.clientSocket.recv(self.base_Buffsize)
                    data=self.clientSocket.recv(pkl.loads(databytes_len))
                    if data == pkl.dumps('ENDPORT'):
                        return 'ENDPORT'
                    else:
                        try:
                            self.data_queue.append(pkl.loads(data))
                            self.send('SUCCESS')
                            if self.verbose:
                                print(f'Received data from client {self.clientAddr} with {len(data)} bytes.')
                        except:
                            print('Data received but not pickleable.')
                            self.send('FAIL')
                except IOError as e:
                    print(e)
                    self.clientSocket.close()
                    break
    def close(self):
        if self.verbose:
            print('Closing server...')
        self.clientSocket.close()
        self.tcpSocket.close()