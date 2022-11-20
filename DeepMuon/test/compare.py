'''
Author: airscker
Date: 2022-10-05 21:51:59
LastEditors: airscker
LastEditTime: 2022-11-14 12:33:32
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from DeepMuon.tools.AirFunc import *
from DeepMuon.test.analysis import *

def loss_com(work_dirs,output='',max=0.2,num=0):
    tsl=[]
    trl=[]
    name=[]
    # max=np.arcsin(max)*180/np.pi
    assert os.path.isdir(output),f'Please specify the root path of output,rather then {output}'
    for i in range(len(work_dirs)):
        logfile=os.path.join(work_dirs[i],'log.log')
        if os.path.exists(logfile):
            name.append(work_dirs[i].split('/')[-1])
            train_data=load_log(logfile)
            if num>0:
                train_data=train_data[:num]
            ts=train_data[:,1]
            tr=train_data[:,2]
            if np.min(tr)>10:
                ts/=10000
                tr/=10000
            # ts=np.arcsin(ts)*180/np.pi
            # tr=np.arcsin(tr)*180/np.pi
            ts[ts>max]=max
            tr[tr>max]=max
            tsl.append(ts)
            trl.append(tr)
            print(f'{logfile} loaded, training epochs: {len(tsl[-1])}')
    ts_name=os.path.join(output,'test_com.jpg')
    tr_name=os.path.join(output,'train_com.jpg')
    plot_curve(data=tsl,title='Compare testing performance of several models',data_label=name,save=ts_name)
    plot_curve(data=trl,title='Compare training performance of several models',data_label=name,save=tr_name)
    print(f'The compare curve is saved as:\nTraining: {tr_name}\nTesting: {ts_name}\nCompared Model work_dirs: {name}')
    return tsl,trl,name

@click.command()
@click.option('--root',default='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV')
@click.option('--max',default=0.2)
@click.option('--num',default=0)
def run(root,max,num):
    # work_dirs=os.listdir(root)
    # work_dirs=['CSPP_4','CSPP_5','DResMax_1','ResMax_4']
    work_dirs=['CSPP_5','ResMax_5','DResMax_1','DResMax_2']
    
    # work_dirs=['MLP3_3D','MLP3_3D2','MLP3_3D3','MLP3_3D4','UNET_MLP','UNET_MLP_2','UNET_MLP_D','CNN1','CNN2','CNN3','CNN4']
    for i in range(len(work_dirs)):
        work_dirs[i]=os.path.join(root,work_dirs[i])
        loss_com(work_dirs,root,max,num=num)
        # info=load_log(os.path.join(root,work_dirs[i],'log.log'))
        # info=np.sqrt(3*info)*180/np.pi
        # info=np.arcsin(info)*180/np.pi
        # info[info>16]=16
        # plot_curve(data=[info[:,2],info[:,1]],title='ResMax',axis_label=['Epoch','Angle loss(degrees)'],data_label=['Training','Testing'],save=os.path.join(root,work_dirs[i],'loss_curve.jpg'))
        


if __name__=='__main__':
    print('\n---Comparing...---')
    run()
    
    