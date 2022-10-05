'''
Author: airscker
Date: 2022-10-05 15:08:10
LastEditors: airscker
LastEditTime: 2022-10-05 17:51:48
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
from email.policy import default
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

from DeepMuon.tools.AirFunc import *
from DeepMuon.tools.AirConfig import Config
from DeepMuon.tools.AirLogger import LOGT

def loss_dist(loss,title='Loss',bins=15,sigma=3,save='',show=False):
    plot_hist_2nd(loss,title=title,bins=bins,sigma=sigma,save=save,show=show)

def thres_dist(loss,thres,title='Loss',bins=15,sigma=3,save='',show=False):
    loss=np.array(loss)
    loss=loss[loss<thres]
    plot_hist_2nd(loss,title=title,bins=bins,sigma=sigma,save=save,show=show)

def data_analysis(thres,sigma,config):
    assert os.path.exists(config),f'Config file {config} can not be found'
    train_config=Config(configpath=config)
    work_dir=train_config.paras['work_config']['work_dir']
    res_path=os.path.join(work_dir,'infer','inference_res.pkl')
    assert os.path.exists(res_path),f'Result file {res_path} can not be found'
    with open(res_path,'rb')as f:
        res=pkl.load(f)
    f.close()
    loss=np.array(res['loss'])
    logger=LOGT(log_dir=os.path.join(work_dir,'ana'),logfile='ana.log',new=False)
    logger.log(f'================= Current Time: {time.ctime()} =================')
    logger.log(f'Threshold: {thres}')
    logger.log(f'Sigma: {sigma}')
    name=os.path.join(work_dir,'ana',f'loss_sigma_{sigma}.jpg')
    loss_dist(loss,save=name)
    logger.log(f'Distribution Histogram Image of Loss value is saved as {name}')
    name=os.path.join(work_dir,'ana',f'loss_sigma_{sigma}_thres_{thres}.jpg')
    thres_dist(loss,thres=thres,save=name)
    logger.log(f'Distribution Histogram Image of Loss value Threshold {thres} is saved as {name}')
    sigma_range=[np.mean(loss)-sigma*np.std(loss),np.mean(loss)+sigma*np.std(loss)]
    thres_loss=loss[loss<thres]
    sigma_range_thres=[np.mean(thres_loss)-sigma*np.std(thres_loss),np.mean(thres_loss)+sigma*np.std(thres_loss)]
    logger.log(f'{sigma}Sigma range: {sigma_range}')
    logger.log(f'{sigma}Sigma range of threshold value {thres}: {sigma_range_thres}')
    logger.log(f'Total number of samples: {len(loss)}')
    logger.log(f'Total number of threshold {thres} resampled samples: {len(thres_loss)}')
    

@click.command()
@click.option('--thres',default=0.005)
@click.option('--sigma',default=3)
@click.option('--config',default='')
def run(thres,sigma,config):
    data_analysis(thres,sigma,config)

if __name__=='__main__':
    print('\n---Starting Analysis...---')
    run()
