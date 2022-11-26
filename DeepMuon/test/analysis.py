'''
Author: airscker
Date: 2022-10-05 15:08:10
LastEditors: airscker
LastEditTime: 2022-11-26 12:27:31
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

def load_log(log_file):
    """Loads the training log from the given log file .

    Args:
        log_file ([type]): The path of the log file.
    
    Return:
        [epoch:[lr,tsl,trl,btsl]]

    """
    assert os.path.exists(log_file),f'Training log {log_file} can not be found'
    with open(log_file,'r')as f:
        info=f.readlines()
    train_info=[]
    for i in range(len(info)):
        info[i]=info[i].split('\n')[0]
        if 'LR' in info[i] and 'Test Loss' in info[i] and 'Train Loss' in info[i] and 'Best Test Loss' in info[i]:
            data=info[i].split(',')
            epoch=int(data[1].split('[')[1].split(']')[0])
            train_data=[float(data[0].split(': ')[-1]),float(data[2].split(': ')[-1]),float(data[3].split(': ')[-1]),float(data[4].split(': ')[-1])]
            if epoch>len(train_info):
                train_info.append(train_data)
            else:
                train_info[epoch-1]=train_data
    return np.array(train_info)

def data_analysis(thres,sigma,config,fold=0.8):
    """
    The data_analysis function is used to analyze the distribution of loss value and plot the loss/lr curve.
    The function will also log key information of loss, including mean, std, sigma range and threshold value sigma range.
    
    
    :param thres: Determine the threshold value of loss
    :param sigma: Set the threshold of loss value
    :param config: Specify the configuration file of the training process
    :param fold=0.8: Set the threshold of loss value
    :return: The result of the loss value distribution and threshold distribution, as well as the loss/lr curve
    """
    # Load configuration
    assert os.path.exists(config),f'Config file {config} can not be found'
    train_config=Config(configpath=config)
    work_dir=train_config.paras['work_config']['work_dir']
    res_path=os.path.join(work_dir,'infer','inference_res.pkl')
    assert os.path.exists(res_path),f'Result file {res_path} can not be found'
    # Load inferenced results
    with open(res_path,'rb')as f:
        res=pkl.load(f)
    f.close()
    # Plot distibution of loss value
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
    # Plot loss/lr curve
    train_data=load_log(os.path.join(work_dir,train_config.paras['work_config']['logfile']))
    test_loss=train_data[:,1]
    train_loss=train_data[:,2]
    fold_loss=np.max(test_loss)-fold*(np.max(test_loss)-np.min(test_loss))
    fold_index=np.where(test_loss<=(fold_loss))[0][0]
    curve_path=os.path.join(work_dir,'ana',f'TR_TS_loss.jpg')
    plot_curve(data=[test_loss,train_loss],title='Loss Value Curve',data_label=['Test Loss','Train Loss'],save=curve_path)
    logger.log(f'Loss Value Curve is saved as {curve_path}')
    curve_path=os.path.join(work_dir,'ana',f'TR_TS_loss_f_{fold_index}.jpg')
    plot_curve(data=[test_loss[fold_index:],train_loss[fold_index:]],title=f'Loss Value Curve Start From Fold_{fold} Index: {fold_index}',data_label=['Test Loss','Train Loss'],save=curve_path)
    logger.log(f'Loss Value Curve Start From Fold_{fold} Index: {fold_index} is saved as {curve_path}')
    curve_path=os.path.join(work_dir,'ana','lr_curve.jpg')
    plot_curve(data=train_data[:,0],title='Learn Rate',data_label=['Learn Rate'],axis_label=['Epoch','LR'],save=curve_path)
    logger.log(f'Learn rate curve during training is saved as {curve_path}')
    # Log key information of loss
    sigma_range=[np.mean(loss)-sigma*np.std(loss),np.mean(loss)+sigma*np.std(loss)]
    thres_loss=loss[loss<thres]
    sigma_range_thres=[np.mean(thres_loss)-sigma*np.std(thres_loss),np.mean(thres_loss)+sigma*np.std(thres_loss)]
    logger.log(f'{sigma}Sigma range: {sigma_range}')
    logger.log(f'{sigma}Sigma range of threshold value {thres}: {sigma_range_thres}')
    logger.log(f'Total number of samples: {len(loss)}')
    logger.log(f'Total number of threshold {thres} resampled samples: {len(thres_loss)}')
    

@click.command()
@click.option('--thres',default=0.003)
@click.option('--sigma',default=3)
@click.option('--config',default='')
@click.option('--fold',default=0.8)
def run(thres,sigma,config,fold):
    data_analysis(thres,sigma,config,fold)

if __name__=='__main__':
    print('\n---Starting Analysis...---')
    run()
