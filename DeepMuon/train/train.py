'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-01-31 12:39:48
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import time
import os
from tqdm import tqdm
import click

from DeepMuon.tools.AirConfig import Config
import DeepMuon.tools.AirFunc as AirFunc
import DeepMuon.tools.AirLogger as AirLogger

import torch
from torch import nn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)


def main(config_info, msg=''):
    '''Initialize the basic training configuration'''
    configs = config_info.paras
    batch_size = configs['hyperpara']['batch_size']
    epochs = configs['hyperpara']['epochs']
    train_data = configs['train_dataset']['params']
    test_data = configs['test_dataset']['params']
    work_dir = configs['work_config']['work_dir']
    log = configs['work_config']['logfile']
    resume = configs['checkpoint_config']['resume_from']
    load = configs['checkpoint_config']['load_from']
    inter = configs['checkpoint_config']['save_inter']
    gpu = configs['gpu_config']['gpuid']
    logger = AirLogger.LOGT(log_dir=work_dir, logfile=log)
    json_logger = AirLogger.LOGJ(log_dir=work_dir, logfile=f'{log}.json')
    '''Create work_dir'''
    try:
        os.makedirs(work_dir)
    except:
        pass
    log = os.path.join(work_dir, log)
    '''show hyperparameters'''
    logger.log(
        f'========= Current Time: {time.ctime()} Current PID: {os.getpid()} =========')
    if not os.path.exists(msg):
        logger.log('LICENSE MISSED! REFUSE TO START TRAINING')
        return 0
    with open(msg, 'r')as f:
        msg = f.read()
    logger.log(msg)
    logger.log(config_info)

    '''
    Load datasets
    eg. train_dataset=PandaxDataset(IMG_XY_path=train_data)
        test_dataset=PandaxDataset(IMG_XY_path=test_data)
    In the example shown above, `configs['train_dataset']['backbone']` <> `PandaxDataset`, `IMG_XY_path=train_data` <> `**train_data`
    '''
    train_dataset = configs['train_dataset']['backbone'](**train_data)
    test_dataset = configs['test_dataset']['backbone'](**test_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    ''' Get cpu or gpu device for training.'''
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using {device} device")

    '''
    Create Model and optimizer/loss/scheduler
    You can change the name of net as any you want just make sure the model structure is the same one
    eg. model = MLP3().to(device)
    In the example shown above, `MLP3` <> `configs['model']['backbone']`, `model_parameters` <> `**configs['model']['params']`
    '''
    model: nn.Module = configs['model']['backbone'](
        **configs['model']['params']).to(device)
    epoch_now = 0
    if resume == '' and load == '':
        pass
    elif resume != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
            path=resume, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = epoch_c+1
        logger.log(f'Model Resumed from {resume}, Epoch now: {epoch_now}')
    elif load != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
            path=load, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = 0
        logger.log(
            f'Pretrained Model Loaded from {load}, Epoch now: {epoch_now}')
    epochs += epoch_now
    model_name = model._get_name()
    '''save model architecture'''
    writer = SummaryWriter(os.path.join(work_dir, 'LOG'))
    writer.add_graph(model, torch.randn(
        configs['hyperpara']['inputshape']).to(device))
    '''
    Initialize loss/optimizer/scheduler
    eg. loss_fn=nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
    In the example shown above:
        `nn.MSELoss` <> `configs['loss_fn']['backbone']`, `loss_function parameters` <> `**configs['loss_fn']['params']`
        `torch.optim.SGD` <> `configs['optimizer']['backbone']`, `lr=0.001, momentum=0.9, nesterov=True` <> `**configs['optimizer']['params']`
        `torch.optim.lr_scheduler.ReduceLROnPlateau` <> `configs['scheduler']['backbone']`, `mode='min', factor=0.5, patience=100` <> `**configs['scheduler']['params']`
    '''
    loss_fn = configs['loss_fn']['backbone'](**configs['loss_fn']['params'])
    optimizer = configs['optimizer']['backbone'](
        model.parameters(), **configs['optimizer']['params'])
    scheduler = configs['scheduler']['backbone'](
        optimizer, **configs['scheduler']['params'])

    '''Log the information of the model'''
    flops, params = get_model_complexity_info(model, tuple(
        configs['hyperpara']['inputshape']), as_strings=True, print_per_layer_stat=False, verbose=True)
    logger.log(f'GFLOPs: {flops}, Number of Parameters: {params}')
    logger.log(f'Model Architecture:\n{model}')
    logger.log(f'Loss Function: {loss_fn}')
    logger.log(f'Optimizer:\n{optimizer}')
    logger.log(
        f'scheduler: {scheduler.__class__.__name__}:\n\t{scheduler.state_dict()}')

    '''Training Initailization'''
    bar = tqdm(range(epoch_now, epochs), mininterval=1)
    bestloss = test(device, test_dataloader, model, loss_fn)
    for t in bar:
        start_time = time.time()
        trloss = train(device, train_dataloader, model,
                       loss_fn, optimizer, scheduler)
        tsloss = test(device, test_dataloader, model, loss_fn)
        LRn = optimizer.state_dict()['param_groups'][0]['lr']
        bar.set_description(
            f'LR: {LRn},Test Loss: {tsloss},Train Loss: {trloss}')
        writer.add_scalar(f'Test Loss Curve', tsloss, global_step=t+1)
        writer.add_scalar(f'Train Loss Curve', trloss, global_step=t+1)
        if tsloss <= bestloss:
            bestloss = tsloss
            '''Double save to make sure secure, directly save total model is forbidden, otherwise load issues occur'''
            savepath = os.path.join(work_dir, 'Best_Performance.pth')
            AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                               loss_fn=loss_fn, scheduler=scheduler, path=savepath)
            AirFunc.save_model(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                               scheduler=scheduler, path=os.path.join(work_dir, f'{model_name}_Best_Performance.pth'))
            logger.log(
                f'Best Model Saved as {savepath}, Best Test Loss: {bestloss}, Current Epoch: {(t+1)}', show=False)
        if (t+1) % inter == 0:
            savepath = os.path.join(work_dir, f'Epoch_{t+1}.pth')
            AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                               loss_fn=loss_fn, scheduler=scheduler, path=savepath)
            logger.log(
                f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=False)
        epoch_time = time.time()-start_time
        eta = AirFunc.format_time((epochs-1-t)*(time.time()-start_time))
        mem_info = get_mem_info()
        logger.log(
            f"LR: {LRn}, Epoch: [{t+1}][{epochs}], Test Loss: {tsloss}, Train Loss: {trloss}, Best Test Loss: {bestloss}, Time:{epoch_time}s, ETA: {eta}, Memory Left: {mem_info['mem_left']} Memory Used: {mem_info['mem_used']}", show=False)
        json_logger.log(dict(mode='train', lr=LRn, epoch=t+1, total_epoch=epochs, test_loss=tsloss,
                             train_loss=trloss, best_test_loss=bestloss, time=epoch_time, eta=eta, memory_left=mem_info['mem_left'], memory_used=mem_info['mem_used']))
    return bestloss


def get_mem_info():
    gpu_id = torch.cuda.current_device()
    mem_total = torch.cuda.get_device_properties(gpu_id).total_memory
    mem_cached = torch.cuda.memory_reserved(gpu_id)
    mem_allocated = torch.cuda.memory_allocated(gpu_id)
    return dict(mem_left=f"{(mem_total-mem_cached-mem_allocated)/1024**2:0.2f} MB", total_mem=f"{mem_total/1024**2:0.2f} MB", mem_used=f"{(mem_cached+mem_allocated)/1024**2:0.2f} MB")


def train(device, dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    train_loss = 0
    batchs = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        '''Compute prediction error'''
        pred = model(X)
        loss = loss_fn(pred, y)
        '''Backpropagation'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    # scheduler.step(train_loss/batchs)
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
        main(train_config, msg)
    else:
        print('Distributed Training is not supported!')


if __name__ == '__main__':
    print(f'\n---Starting Neural Network...PID:{os.getpid()}---')
    run()
