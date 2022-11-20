'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: error: git config user.name & please set dead value or install git
LastEditTime: 2022-11-20 09:45:59
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import time
import os
from tqdm import tqdm
import click
# import nni

from DeepMuon.tools.AirConfig import Config
from DeepMuon.tools.AirFunc import load_model, save_model, format_time
from DeepMuon.tools.AirLogger import LOGT

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from ptflops import get_model_complexity_info
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
# @click.command()
# @click.option('--batch_size',default=400)
# @click.option('--epochs',default=10)
# @click.option('--train_data',default='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl')
# @click.option('--test_data',default='./Hailing-Muon/data/1TeV/Hailing_1TeV_val_data.pkl')
# @click.option('--lr',default=0.0005)
# @click.option('--resume',default='')
# @click.option('--load',default='')
# @click.option('--patience',default=100)
# @click.option('--momentum',default=0.9)
# @click.option('--gpu',default=0)
# @click.option('--log',default='log.log')
# @click.option('--work_dir',default=f'./Hailing-Muon/work_dir/1TeV/MLP3_3D')
# @click.option('--inter',default=10,help='Interval of model saving')
# def main(batch_size,epochs,train_data,test_data,lr,lr_step,resume,momentum,gpu,log):
# def main(batch_size,epochs,train_data,test_data,lr,patience,resume,load,momentum,gpu,log,work_dir,inter):


def main(configs, msg=''):
    # Initialize the basic training configuration
    batch_size = configs['hyperpara']['batch_size']
    epochs = configs['hyperpara']['epochs']
    train_data = configs['train_dataset']['params']
    test_data = configs['test_dataset']['params']
    work_dir = configs['work_config']['work_dir']
    log = configs['work_config']['logfile']
    patience = configs['lr_config']['patience']
    lr = configs['lr_config']['init']
    resume = configs['checkpoint_config']['resume_from']
    load = configs['checkpoint_config']['load_from']
    inter = configs['checkpoint_config']['save_inter']
    gpu = configs['gpu_config']['gpuid']
    logger = LOGT(log_dir=work_dir, logfile=log)
    # Create work_dir
    try:
        os.makedirs(work_dir)
    except:
        pass
    log = os.path.join(work_dir, log)

    # show hyperparameters
    logger.log(f'========= Current Time: {time.ctime()} =========')
    logger.log(f'Current PID: {os.getpid()}')
    if not os.path.exists(msg):
        logger.log('LICENSE MISSED! REFUSE TO START TRAINING')
        return 0
    with open(msg, 'r')as f:
        msg = f.read()
    logger.log(msg)
    keys = list(configs.keys())
    info = ''
    for i in range(len(keys)):
        info += f'\n{keys[i]}:'
        info_keys = list(configs[keys[i]].keys())
        for j in range(len(info_keys)):
            info += f'\n\t{info_keys[j]}: {configs[keys[i]][info_keys[j]]}'
    logger.log(info)
    # logger.log(f'Batch Size: {batch_size}, Epochs: {epochs}, LR patience Step: {patience}, Initial Learn Rate: {lr}, GPUID: {gpu}, Resume From: {resume} Load from: {load}')
    # logger.log(f'Command: --batch_size={batch_size} --epochs={epochs} --resume="{resume}" --load={load} --patience={patience} --gpu={gpu} --lr={lr} --work_dir="{work_dir}" --inter="{inter}"')

    # load datasets
    # train_dataset=PandaxDataset(IMG_XY_path=train_data)
    # test_dataset=PandaxDataset(IMG_XY_path=test_data)
    train_dataset = configs['train_dataset']['backbone'](**train_data)
    test_dataset = configs['test_dataset']['backbone'](**test_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Get cpu or gpu device for training.
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using {device} device")

    # Create Model and optimizer/loss/schedular
    # You can change the name of net as any you want just make sure the model structure is the same one
    model = configs['model']['backbone'](
        **configs['model']['params']).to(device)
    epoch_now = 0
    if resume == '' and load == '':
        pass
    elif resume != '':
        epoch_c, model_c, optimizer_c, schedular_c, loss_fn_c = load_model(
            path=resume, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = epoch_c+1
        logger.log(f'Model Resumed from {resume}, Epoch now: {epoch_now}')
    elif load != '':
        epoch_c, model_c, optimizer_c, schedular_c, loss_fn_c = load_model(
            path=load, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = 0
        logger.log(
            f'Pretrained Model Loaded from {load}, Epoch now: {epoch_now}')
    epochs += epoch_now
    model_name = model._get_name()
    # loss/optimizer/lr
    # loss_fn=nn.MSELoss()
    loss_fn = configs['loss_fn']['backbone'](configs['loss_fn']['params'])
    # loss_fn=MSALoss()
    # loss_fn=nn.L1Loss()

    # MLP3_pretrain
    # optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    # MLP3_2
    # optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.1)
    # MLP3_3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # schedular=torch.optim.lr_scheduler.StepLR(optimizer,lr_step,gamma=0.5)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience)
    # Get GFLOPS of the model
    # shape of the input data
    flops, params = get_model_complexity_info(model, tuple(
        configs['hyperpara']['inputshape']), as_strings=True, print_per_layer_stat=False, verbose=True)
    logger.log(f'GFLOPs: {flops}, Number of Parameters: {params}')
    logger.log(f'Model Architecture:\n{model}')
    logger.log(f'Loss Function: {loss_fn}')
    logger.log(f'Optimizer:\n{optimizer}')
    logger.log(f'Schedular: {schedular}')

    # save model architecture
    writer = SummaryWriter(os.path.join(work_dir, 'LOG'))
    writer.add_graph(model, torch.randn(
        configs['hyperpara']['inputshape']).to(device))

    # start training
    bar = tqdm(range(epoch_now, epochs), mininterval=1)
    bestloss = test(device, test_dataloader, model, loss_fn)
    for t in bar:
        start_time = time.time()
        tloss = train(device, train_dataloader, model,
                      loss_fn, optimizer, schedular)
        loss = test(device, test_dataloader, model, loss_fn)
        LRn = optimizer.state_dict()['param_groups'][0]['lr']
        bar.set_description(f'LR: {LRn},Test Loss: {loss},Train Loss: {tloss}')
        writer.add_scalar(f'Test Loss Curve', loss, global_step=t+1)
        writer.add_scalar(f'Train Loss Curve', tloss, global_step=t+1)
        # nni.report_intermediate_result(loss)
        if loss <= bestloss:
            bestloss = loss
            # Double save to make sure secure, directly save total model is forbidden, otherwise load issues occur
            savepath = os.path.join(work_dir, 'Best_Performance.pth')
            save_model(epoch=t, model=model, optimizer=optimizer,
                       loss_fn=loss_fn, schedular=schedular, path=savepath)
            save_model(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                       schedular=schedular, path=os.path.join(work_dir, f'{model_name}_Best_Performance.pth'))
            logger.log(
                f'Best Model Saved as {savepath}, Best Test Loss: {bestloss}, Current Epoch: {(t+1)}', show=False)
        if (t+1) % inter == 0:
            # torch.save(model,os.path.join(work_dir,f'Epoch_{t+1}.pth'))
            savepath = os.path.join(work_dir, f'Epoch_{t+1}.pth')
            save_model(epoch=t, model=model, optimizer=optimizer,
                       loss_fn=loss_fn, schedular=schedular, path=savepath)
            logger.log(
                f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=False)
        logger.log(
            f'LR: {LRn}, Epoch: [{t+1}][{epochs}], Test Loss: {loss}, Train Loss: {tloss}, Best Test Loss: {bestloss}, Time:{time.time()-start_time}s, ETA: {format_time((epochs-1-t)*(time.time()-start_time))}', show=False)
    # nni.report_final_result(loss)
    return bestloss


def train(device, dataloader, model, loss_fn, optimizer, schedular):
    model.train()
    train_loss = 0
    batchs = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # schedular.step()
    schedular.step(train_loss/batchs)
    return train_loss/batchs


def test(device, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss
# nniparams={'lr':0.001}
# optimized_params=nni.get_next_parameter()
# nniparams.update(optimized_params)


@click.command()
@click.option('--config', default='/home/dachuang2022/Yufeng/DeepMuon/config/Hailing/SCSPP.py')
@click.option('--msg', default='')
def run(config, msg):
    train_config = Config(configpath=config)
    if train_config.paras['gpu_config']['distributed'] == False:
        train_config.paras['config'] = {'path': config}
        main(train_config.paras, msg)
    else:
        print('Distributed Training is not supported!')


if __name__ == '__main__':
    print('\n---Starting Neural Network...---')
    run()
