'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2022-10-05 18:43:17
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import time
import os
from tqdm import tqdm
import click
import warnings
import numpy as np
import pickle as pkl

from DeepMuon.tools.AirConfig import Config
from DeepMuon.tools.AirFunc import load_model,format_time,plot_hist_2nd
from DeepMuon.tools.AirLogger import LOGT
from DeepMuon.test.model_info import model_para
from DeepMuon.test.analysis import loss_dist,data_analysis

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from ptflops import get_model_complexity_info
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark = True

def main(configs,ana,thres):
    # Initialize the basic training configuration
    loss_fn=configs['loss_fn']['backbone']()
    batch_size=1
    test_data=configs['test_dataset']['datapath']
    work_dir=configs['work_config']['work_dir']
    assert os.path.exists(work_dir),f'The work_dir specified in the config file can not be found: {work_dir}'
    log='inference.log'
    infer_path=os.path.join(work_dir,'infer')
    res=os.path.join(infer_path,'inference_res.pkl')
    load=os.path.join(work_dir,'Best_Performance.pth')
    gpu=configs['gpu_config']['gpuid']
    logger=LOGT(log_dir=infer_path,logfile=log,new=True)
    log=os.path.join(infer_path,log)
    ana_path=os.path.join(work_dir,'ana')

    # load datasets
    test_dataset=configs['test_dataset']['backbone'](test_data)
    test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    
    # Get cpu or gpu device for training.
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # show hyperparameters
    logger.log(f'========= Current Time: {time.ctime()} =========')
    logger.log(f'Current PID: {os.getpid()}')
    config_path=configs['config']['path']
    logger.log(f'Configuration loaded from {config_path}')
    logger.log(f'Batch size is set as {batch_size} during model inference')
    logger.log(f'Test Dataset loaded from {test_data}')
    logger.log(f'Device Used: {device}')
    logger.log(f'Work_dir of the model: {work_dir}')
    logger.log(f'Inference results will be save into work_dir of the model\nLog info will be saved into {log}')
    logger.log(f'All inference results will be saved into {res}')
    if ana:
        logger.log(f'Data analysis results will be saved into {ana_path}')
    

    # You can change the name of net as any you want just make sure the model structure is the same one
    model = configs['model']['backbone']().to(device)
    assert os.path.exists(load),f'Model inferenced can not be found: {load}'
    epoch_c,model_c,optimizer_c,schedular_c,loss_fn_c=load_model(path=load,device=device)
    model.load_state_dict(model_c,False)
    model.to(device)
    logger.log(f'Pretrained Model Loaded from {load}')
    model_name=model._get_name()
    
    # Get GFLOPS of the model
    flops,params,sumres=model_para(model=configs['model']['backbone'],datasize=configs['hyperpara']['inputshape'],depth=5,gpu=gpu)
    logger.log(f'Model Architecture:\n{model}')
    logger.log(f'{sumres}',show=False)
    logger.log(f'GFLOPs: {flops}, Number of Parameters: {params}')
    logger.log(f'Loss Function: {loss_fn}')

    # save model architecture
    # writer=SummaryWriter(os.path.join(work_dir,'LOG'))
    # writer.add_graph(model,torch.rand(configs['hyperpara']['inputshape']).to(device))

    # start inferencing
    loss,pred,real,loss_map=test(device,test_dataloader, model, loss_fn,logger,thres=thres,ana=ana)
    # Save results
    with open(res,'wb')as f:
        pkl.dump({'loss':loss,'pred':pred,'real':real},f)
    f.close()
    logger.log(f'Inference Results Saved in {res}\nAvailable Results: dict_keys(loss pred real)')
    # Analysis the results
    if ana==True:
        if not os.path.exists(ana_path):
            os.makedirs(ana_path)
        # loss=np.array(loss)
        # name=os.path.join(ana_path,'loss_dist.jpg')
        # loss_dist(loss,save=name)
        data_analysis(thres,sigma=3,config=config_path)
        logger.log(f'Distribution Histogram Image of Loss value is saved into {ana_path}')
        logger.log(f'The loss threshold is set as {thres}\nLoss MEAN STD: {np.mean(loss)} {np.std(loss)}')
        name=os.path.join(ana_path,f'loss_thres_{thres}.pkl')
        with open(name,'wb')as f:
            pkl.dump(loss_map,f)
        f.close()
        logger.log(f'Threshold {thres} Loss ID-[data,pred,real,loss] hash map saved as {name}, total number: {len(loss_map)}')


def test(device,dataloader, model, loss_fn, logger:LOGT,thres,ana=True):
    num_batches = len(dataloader)
    model.eval()
    test_loss=[]
    pred_value=[]
    real_value=[]
    num=1
    loss_map={}
    with torch.no_grad():
        for X, y in dataloader:
            start_time=time.time()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_value=loss_fn(pred, y).item()
            test_loss.append(loss_value)
            pred_value.append(pred.cpu().numpy()[0])
            real_value.append(y.cpu().numpy()[0])
            if ana and loss_value>thres:
                loss_map[num-1]=[X.cpu().numpy(),pred_value[num-1],real_value[num-1],loss_value]
            now_time=time.time()
            logger.log(f'{num}/{num_batches} Loss: {loss_value}, Predicted: {pred_value[num-1]}, Real: {real_value[num-1]}, Time: {now_time-start_time}s, ETA: {format_time((num_batches-num)*(now_time-start_time))}')
            num+=1
    return test_loss,pred_value,real_value,loss_map

@click.command()
@click.option('--config',default='/home/dachuang2022/Yufeng/DeepMuon/config/Hailing/Vit.py')
@click.option('--ana',default=True)
@click.option('--thres',default=0.004)
def run(config,ana,thres):
    train_config=Config(configpath=config)
    if train_config.paras['gpu_config']['distributed']==True:
        warnings.warn('Distributed Training is not supported during model inference')
    train_config.paras['config']={'path':config}
    main(train_config.paras,ana,thres)
        

if __name__=='__main__':
    print('\n---Starting Neural Network...---')
    run()