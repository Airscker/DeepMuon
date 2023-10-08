'''
Author: airscker
Date: 2023-10-06 22:42:15
LastEditors: airscker
LastEditTime: 2023-10-08 02:42:49
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import click
import socket
import ctypes
import time
import logging
import torch
import pickle as pkl
from multiprocessing.managers import SharedMemoryManager

import DeepMuon
from DeepMuon.tools.AirServer.shared_memory import SharedMemory
from DeepMuon.tools.AirFunc import fix_port,ddp_fsdp_model_save
from DeepMuon.tools.AirQuene import TaskFIFOQueueThread
from DeepMuon.tools.AirLogger import LOGT
pkg_path=DeepMuon.__path__[0]

class DatasetServer(object):
    def __init__(self,
                 serverIP:str='127.0.0.1',
                 serverPort:int=11300,
                 listenNum:int=1,
                 verbose:bool=False,
                 workdir:str='',
                 logfile:str='FileSavingServer.log'):
        self.BestModelFIFOSaving=TaskFIFOQueueThread(sequenced=False,verbose=verbose,daemon=True)
        self.CheckpointFIFOSaving=TaskFIFOQueueThread(sequenced=True,verbose=verbose,daemon=True)
        self.BestModelFIFOSaving.start()
        self.CheckpointFIFOSaving.start()
        self.serverIP = serverIP
        available_port,info= fix_port(serverIP,serverPort)
        self.serverPort = available_port
        self.verbose = verbose
        self.log_savepath=os.path.join(workdir,'Server')
        self.logger=LOGT(self.log_savepath,logfile)
        self.base_Buffsize = 1024
        self.ADDR = (self.serverIP, self.serverPort)
        
        self.log('*'*50)
        self.log(f'Initializing Server, PID: {os.getpid()}')
        self.log(info)
        self.tcpSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpSocket.bind(self.ADDR)
        self.tcpSocket.listen(listenNum)
        self.log(f'Prepared to receive data from {self.ADDR}')
        self.log(f'Number of clients to listen: {listenNum}')
        self.log(f'Base Buffsize: {self.base_Buffsize}')
        with open(os.path.join(self.log_savepath,'ServerInfo.pkl'),'wb') as f:
            pkl.dump(self.ADDR,f)

        self.data_queue = []
        end_signal=self.receive()
        if end_signal == 'ENDPORT':
            self.log('Received ENDPORT signal. Closing server...')
            self.close()
    def log(self,msg):
        if msg is not None and msg != '':
            self.logger.log(msg,show=self.verbose,timer=True)
            # if self.verbose:
            #     print(msg)
    def send(self, data):
        data_bytes=data.encode('utf-8')
        self.log(f'Sending data to {self.clientAddr} with {len(data_bytes)} bytes: {data}')
        self.clientSocket.send(data_bytes)
        self.log('Data sent.')
    def receive(self):
        while True:
            self.log(f'Waiting for connection...')
            self.clientSocket, self.clientAddr = self.tcpSocket.accept()
            self.log(f'Connected from client: {self.clientAddr}')
            while True:
                try:
                    data_bytes=self.clientSocket.recv(self.base_Buffsize)
                    data=data_bytes.decode('utf-8')
                    if data == 'ENDPORT':
                        return 'ENDPORT'
                    else:
                        try:
                            self.data_queue.append(data)
                            self.log(f'Received shared memory space from client {self.clientAddr} with {len(data_bytes)} bytes: {data}')
                            shm=SharedMemory(name=data)
                            model_type,kwargs=pkl.loads(shm.buf)
                            self.log(f'Data decoded from shared memory space: {data}')
                            shm.close()
                            self.send('SUCCESS')
                            self.log(f'Starting to save data with ddp_fsdp_model_save method.')
                            if model_type == 'BestModel':
                                self.BestModelFIFOSaving.add_task(ddp_fsdp_model_save,kwargs,len(self.data_queue))
                                self.log(f'BestModel data added to FIFO model saving quene.')
                            elif model_type == 'Checkpoint':
                                self.CheckpointFIFOSaving.add_task(ddp_fsdp_model_save,kwargs,len(self.data_queue))
                                self.log(f'Checkpoint data added to FIFO checkpoint saving quene.')
                            else:
                                self.log(f'Unrecognized model type: {model_type}')
                        except:
                            self.log('Data received but not decodable.')
                            self.send('FAIL')
                except IOError as e:
                    self.log(str(e))
                    self.clientSocket.close()
                    self.log(f'Connection from {self.clientAddr} closed.')
                    break
    def close(self):
        self.BestModelFIFOSaving.end_task()
        self.CheckpointFIFOSaving.end_task()
        self.clientSocket.close()
        self.tcpSocket.close()
        os.remove(os.path.join(self.log_savepath,'ServerInfo.pkl'))
        self.log('Server closed.')
def save_model():
    time.sleep(10)
    torch.save(torch.rand(1000,1000),'./test.pt')

class DatasetClient(object):
    def __init__(self,
                 serverIP:str='127.0.0.1',
                 serverPort:int=11200,
                 verbose:bool=False,
                 workdir:str='',
                 logfile:str='FileSavingClient.log'):
        self.serverIP = serverIP
        self.serverPort = serverPort
        self.verbose = verbose
        log_savepath=os.path.join(workdir,'Server')
        self.logger=LOGT(log_savepath,logfile)
        self.base_Buffsize = 1024
        self.ADDR = (self.serverIP, self.serverPort)
        self.log('*'*50)
        self.log(f'Initializing Client, PID: {os.getpid()}')
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect(self.ADDR)
        self.log(f'Connected to server {self.ADDR}')
        self.log(f'Base Buffsize: {self.base_Buffsize}')
        self.mem_list={}
    def log(self,msg):
        if msg is not None and msg != '':
            self.logger.log(msg,show=self.verbose,timer=True)
            # logging.info(msg)
            # if self.verbose:
            #     print(msg)
    def send(self,data:str):
        '''
        ## Send data to FIFO saving server.

        ### Args:
            - data: data to send, here is the usage tips of `data`:
                - `data` should be the name of the shared memory space which contains a tuple/list with 2 elements, which are `model_type` and `kwargs` respectively.
                    - `model_type` should be a `str` object, which can be either `BestModel` or `Checkpoint`, which indicates the type of the model to be saved,
                        - `model_type` is `BestModel`, the FIFO saving server will save the model checkpoint un-sequencially,
                        - `model_type` is `Checkpoint`, the FIFO saving server will save the model checkpoint sequencially.
                    - `kwargs` should be a `dict` object, which contains the arguments that will be passed to `DeepMuon.tools.ddp_fsdp_model_save`.
            - if `data` is `ENDPORT`, the client will send `ENDPORT` signal to the FIFO saving server, and the FIFO saving server will close itself.
        
        ### Returns:
            - `0` if `data` is `ENDPORT` or the server receivement is failed.
            - `1` if the server receivement is successful.
        '''
        data_bytes=data.encode('utf-8')
        self.clientSocket.send(data_bytes)
        self.log(f'Sent shared memory space name to server {self.ADDR} with {len(data_bytes)} bytes: {data}')
        if data == 'ENDPORT':
            return 0
        response=self.receive()
        if response == 'SUCCESS':
            self.log('Server receivement DONE.')
            return 1
        else:
            self.log('Server receivement FAILED.')
            return 0
    def receive(self):
        # databytes_len=self.clientSocket.recv(self.base_Buffsize)
        # data = self.clientSocket.recv(pkl.loads(databytes_len))
        data_bytes=self.clientSocket.recv(self.base_Buffsize)
        data=data_bytes.decode('utf-8')
        self.log(f'Received data from server {self.ADDR} with {len(data_bytes)} bytes: {data}')
        return data

    def close(self,close_server=False):
        if close_server:
            self.log('Sending ENDPORT signal to server.')
            self.send('ENDPORT')
        self.log('Closing client...')
        self.clientSocket.close()
        self.log('Client closed.')

# if __name__ == '__main__':
#     client = Client(verbose=True)
#     # data=torch.randn(1000,1000)
#     data='client message'
#     for i in range(10):
#         data=torch.randn(1000,1000,1000)
#         print(f'sending message {i}')
#         client.send(str(id(data)))
#         # time.sleep(1)
#         # print(client.receive())
#     client.close(close_server=True)
# @click.command()
# @click.option('--port',default=11300,help='The port number of the server')
# @click.option('--workdir',default='.',help='The working directory of the server')
# def start_filesaving_server(port,workdir):
#     FIFOserver = FileSavingServer(serverPort=port,workdir=workdir,verbose=True)
# if __name__ == '__main__':
#     start_filesaving_server()