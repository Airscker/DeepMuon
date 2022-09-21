'''
Author: airscker
Date: 2022-09-11 08:02:32
LastEditors: airscker
LastEditTime: 2022-09-15 21:32:04
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
from email.policy import default
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import click
import psutil
import os
import GPUtil
import numpy as np
import pynvml
import socket

def get_gpu_info():
    """Get information about all GPUs .
    Returns:
        gpulist: [gpu.id,gpu.memoryTotal,gpu.memoryUsed,gpu.memoryUtil]
        info: gpu info iin HTML format
        pids: list of pids on gpus
    """
    pynvml.nvmlInit()
    Gpus = GPUtil.getGPUs()
    gpulist = []
    pids=[]
    info=f'''<table>{html_addrow(['GPUID','PID','Memory Used(MB)','Total Memory(MB)','Free Memory(MB)','Used Proportion'])}'''
    for gpu in Gpus:
        handle=pynvml.nvmlDeviceGetHandleByIndex(gpu.id)
        pid_list=pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_info=pynvml.nvmlDeviceGetMemoryInfo(handle)
        for i in range(len(pid_list)):
            pids.append(pid_list[i].pid)
            info_list=[gpu.id,pid_list[i].pid,format(pid_list[i].usedGpuMemory/1024**2,'0.2f'),format(gpu_info.total/1024**2,'0.2f'),format(gpu_info.free/1024**2,'0.2f'),format(gpu_info.used/gpu_info.total,'0.2f')]
            info+=html_addrow(info_list)
        gpulist.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed,gpu.memoryUtil * 100])
    gpulist=np.array(gpulist)
    info+='''</table>'''
    pynvml.nvmlShutdown()
    return gpulist,info,pids
    


def html_addrow(msg_list:list):
    """convert a list of messages into a HTML row
    Args:
        msg_list
    Returns:
        message in HTML format
    """
    msg_row='''<tr>'''
    for i in range(len(msg_list)):
        msg_row+=f'''<td>{msg_list[i]}</td>'''
    msg_row+='''</tr>'''
    return msg_row

def get_cpu_info():
    ''' :return:
    memtotal: 总内存
    memfree: 空闲内存
    memused: Linux: total - free,已使用内存
    mempercent: 已使用内存占比
    cpu: 各个CPU使用占比
    '''
    mem = psutil.virtual_memory()
    memtotal = mem.total
    memfree = mem.free
    mempercent = mem.percent
    memused = mem.used
    cpu = psutil.cpu_percent(percpu=True)

    return memtotal, memfree, memused, mempercent, cpu

class Monitor():
    def __init__(self,pid):
        self.pid=pid
        self.info=None
        self.gpu_info=None
        self.gpu_pids=None
        self.last_gpu_info=None
        self.last_gpu_pids=None
        self.__Init()
    def GPU_monitor(self,report=True):
        """TMonitor the GPU memory status and report the change
        Args:
            report: Whether to report the change, Defaults to True.
        """
        print(f'GPU Source Monitor Time: {time.ctime()}')
        self.gpu_info,self.info,self.gpu_pids=get_gpu_info()
        if report==True:
            if np.mean(self.gpu_info[:,-1])<10 and np.mean(self.last_gpu_info[:,-1])>=10:
                self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>GPU Source Monitor Time: {time.ctime()}</h2>\
                    <br>GPU Source Available (AVE {np.mean(self.gpu_info[:,-1]):0.2f} %)\
                    <br>{self.info}','html','utf-8')
                self.__Report()
                print('GPU Source Condition Reported')
            if np.mean(self.gpu_info[:,-1])>=20 and np.mean(self.last_gpu_info[:,-1])<20:
                self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>GPU Source Monitor Time: {time.ctime()}</h2>\
                    <br>GPU Source Occupied (AVE {np.mean(self.gpu_info[:,-1]):0.2f} %)\
                    <br>{self.info}','html','utf-8')
                self.__Report()
                print('GPU Source Condition Reported')
        self.last_gpu_info=self.gpu_info
    def PID_monitor(self,report=True):
        """Monitor the GPU PID status and return the monitored specific PID status
        Args:
            report: Whether to report the status, Defaults to True.
        Returns:
            The status of the specified PID to be monitored
        """
        print(f'PID Status Monitor Time: {time.ctime()}')
        _,self.info,self.gpu_pids=get_gpu_info()
        # Monitor specified PID
        if self.pid is not None:
            try:
                process=psutil.Process(self.pid)
                print(f'{process}<br>Monitor time: {time.ctime()}, Monitor PID: {os.getpid()}, PID watched: {self.pid}')
            except:
                if report==True:
                    self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>PID Monitor Time: {time.ctime()}</h2>\
                        <br>PID Monitor Raise: PID {self.pid} Ended\
                        <br>{self.info}', 'html', 'utf-8')
                    self.__Report()
                    print(f'PID {self.pid} Condition Reported')
                self.pid=None
        # Monitor PID status changes
        terminated_pid=[]
        new_pid=[]
        for i in range(len(self.last_gpu_pids)):
            if self.last_gpu_pids[i] not in self.gpu_pids:
                terminated_pid.append(self.last_gpu_pids[i])
        for i in range(len(self.gpu_pids)):
            if self.gpu_pids[i] not in self.last_gpu_pids:
                new_pid.append(self.gpu_pids[i])
        if report==True:
            if len(terminated_pid)>0 or len(new_pid)>0:
                self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>PID Monitor Time: {time.ctime()}</h2>\
                    <br>PID Monitor Raise: {len(terminated_pid)} PIDs Ended And {len(new_pid)} PIDs Come Up\
                    <br>Terminated PIDs: <br>{terminated_pid}<br>Newly Added PIDs: <br>{new_pid}\
                    <br>Current GPU Info: <br>{self.info}', 'html', 'utf-8')
                self.__Report()
                print(f'PID Operation Condition Reported')
        self.last_gpu_pids=self.gpu_pids
        return self.pid
    def End(self):
        print(f'Monitor End Time: {time.ctime()}')
        self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>Monitor End Time: {time.ctime()}</h2>\
            <br>{self.info}\
            <br>PID Monitor Closed, Monitor PID: {os.getpid()}', 'html', 'utf-8')
        self.__Report()

    def __Init(self):
        self.last_gpu_info,self.info,self.last_gpu_pids=get_gpu_info()
        pid_info=self.PID_monitor(report=False)
        if pid_info is None:
            pid_info='The PID to be monitored does not exist'
        else:
            pid_info=f'The Monitored PID {self.pid} is Running Normally'
        self.message = MIMEText(f'<h2>Server Host: {socket.gethostname()}<br>Monitor Initialization Time: {time.ctime()}</h2>\
            <br>PID Monitor Initialized, Monitor PID: {os.getpid()}\
            <br>{pid_info}\
            <br>{self.info}', 'html', 'utf-8')
        self.__Report()
    def __Report(self):
        mail_host="smtp.qq.com"
        mail_user="3137698717"
        mail_pass="zhswiuascgxzddbc"
        sender = '3137698717@qq.com'
        receivers = ['3137698717@qq.com']
        # message = MIMEText(f'{time.ctime()}\
        #     <br>PID Monitor Raise: {pid} Ended\
        #     <br>{info}', 'html', 'utf-8')
        self.message['From'] = Header(sender, 'utf-8')
        self.message['To'] =  Header(f"{receivers}", 'utf-8')
            
        subject = 'AIMI-Stanford Server PID Monitor Report'
        self.message['Subject'] = Header(subject, 'utf-8')
        try:
            smtpObj = smtplib.SMTP() 
            smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
            smtpObj.login(mail_user,mail_pass)  
            smtpObj.sendmail(sender, receivers, self.message.as_string())
            print (f"Email Sent to {receivers}")
        except smtplib.SMTPException:
            print ("Error!!!")

@click.command()
@click.option('--pid',default=20221266)
@click.option('--delay',default=10)
def main(pid,delay):
    monitor=Monitor(pid=pid)
    for i in range(int(24*7*60*60/delay)):
        # monitor.GPU_monitor()
        monitor.PID_monitor()
        time.sleep(delay)
    monitor.End()
    return 0
if __name__=='__main__':
    main()
    

