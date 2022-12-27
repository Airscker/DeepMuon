'''
Author: airscker
Date: 2022-11-26 23:18:08
LastEditors: airscker
LastEditTime: 2022-12-26 21:52:37
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import pynvml
import GPUtil
import numpy as np
import psutil


def get_gpu_info():
    """Get information about all GPUs .
    Returns:
        gpu_para: [gpu.id,gpu.memoryTotal,gpu.memoryUsed,gpu.memoryUtil]
        info: gpu info iin HTML format
        pids: list of pids on gpus
    """
    pynvml.nvmlInit()
    Gpus = GPUtil.getGPUs()
    gpu_para = []
    pids = []
    info = f'''<table>{html_addrow(['GPUID','PID','Memory Used(MB)','Total Memory(MB)','Free Memory(MB)','Used Proportion'])}'''
    for gpu in Gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.id)
        pid_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        for i in range(len(pid_list)):
            pids.append(pid_list[i].pid)
            info_list = [gpu.id, pid_list[i].pid, format(pid_list[i].usedGpuMemory/1024**2, '0.2f'), format(
                gpu_info.total/1024**2, '0.2f'), format(gpu_info.free/1024**2, '0.2f'), format(gpu_info.used/gpu_info.total, '0.2f')]
            info += html_addrow(info_list)
        gpu_para.append(
            [gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])
    gpu_para = np.array(gpu_para)
    info += '''</table>'''
    pynvml.nvmlShutdown()
    return gpu_para, info, pids


def pid_info(pid):
    """
    The pid_info function returns a list of the following values:
    - pid_info[0] = process name
    - pid_info[2] = number of threads
    - pid_info[3] = status (running, idle, etc.)

    :param pid: Specify the process id
    :return: A list of information about the process
    """

    pid_info = []
    try:
        process = psutil.Process(pid)
    except:
        pass


def html_addrow(msg_list: list):
    """convert a list of messages into a HTML row
    Args:
        msg_list
    Returns:
        message in HTML format
    """
    msg_row = '''<tr>'''
    for i in range(len(msg_list)):
        msg_row += f'''<td>{msg_list[i]}</td>'''
    msg_row += '''</tr>'''
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
