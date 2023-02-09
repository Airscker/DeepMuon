'''
Author: airscker
Date: 2023-02-02 18:30:43
LastEditors: airscker
LastEditTime: 2023-02-09 17:46:12
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''

import time
import os
import click
import numpy as np
import warnings

from DeepMuon.tools.AirConfig import Config
import DeepMuon.tools.AirFunc as AirFunc
import DeepMuon.tools.AirLogger as AirLogger

import torch
from torch import nn
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)


def main(config_info, msg=''):
    '''Initialize the basic training configuration'''
    configs = config_info.paras
    batch_size = configs['hyperpara']['batch_size']
    test_data = configs['test_dataset']['params']
    work_dir = configs['work_config']['work_dir']
    log = 'inference.log'
    resume = configs['checkpoint_config']['resume_from']
    load = configs['checkpoint_config']['load_from']
    assert resume != '' or load != '', 'During inference model checkpoint file (*.pth) expected, however nothing given'
    gpu = configs['gpu_config']['gpuid']
    '''Initialize Distributed Training'''
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    '''Log the basic parameters'''
    logger = AirLogger.LOGT(log_dir=work_dir, logfile=log)
    json_logger = AirLogger.LOGJ(log_dir=work_dir, logfile=f'{log}.json')
    '''Create work_dir'''
    if not os.path.exists(work_dir):
        warnings.warn(
            f"{work_dir} doesn't exists, which will be created automatically.")
        os.makedirs(work_dir)
    log = os.path.join(work_dir, log)
    '''show hyperparameters'''
    logger.log(
        f'========= Current Time: {time.ctime()} Current PID: {os.getpid()} =========')
    config_path = configs['config']['path']
    logger.log(f'Configuration loaded from {config_path}')
    logger.log(f'Batch size is set as {batch_size} during model inference')
    logger.log(f'Test Dataset loaded from {test_data}')
    logger.log(f'Device Used: {device}')
    logger.log(f'Work_dir of the inference: {work_dir}')
    logger.log(
        f'Inference results will be save into work_dir of the model\nLog info will be saved into {log}')
    # if ana:
    #     logger.log(f'Data analysis results will be saved into {ana_path}')
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
    test_dataset = configs['test_dataset']['backbone'](**test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    '''
    Create Model and optimizer/loss/scheduler
    You can change the name of net as any you want just make sure the model structure is the same one
    eg. model = MLP3().to(device)
    In the example shown above, `MLP3` <> `configs['model']['backbone']`, `model_parameters` <> `**configs['model']['params']`
    '''
    model: nn.Module = configs['model']['backbone'](
        **configs['model']['params'])

    checkpoint = resume if resume != '' else load
    epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
        path=checkpoint, device=device)
    model = model.load_state_dict(model_c, False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    logger.log(f'Model loaded from {checkpoint}')
    '''
    Initialize loss/optimizer/scheduler
    eg. loss_fn=nn.MSELoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.1, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=10)
    In the example shown above:
        `nn.MSELoss` <> `configs['loss_fn']['backbone']`, `loss_function parameters` <> `**configs['loss_fn']['params']`
        `torch.optim.SGD` <> `configs['optimizer']['backbone']`, `lr=0.001, momentum=0.9, nesterov=True` <> `**configs['optimizer']['params']`
        `torch.optim.lr_scheduler.ReduceLROnPlateau` <> `configs['scheduler']['backbone']`, `mode='min', factor=0.5, patience=100` <> `**configs['scheduler']['params']`
    '''
    loss_fn = configs['loss_fn']['backbone'](**configs['loss_fn']['params'])

    '''Log the information of the model'''
    # flops, params = get_model_complexity_info(model, tuple(
    #     configs['hyperpara']['inputshape']), as_strings=True, print_per_layer_stat=False, verbose=True)
    # logger.log(f'GFLOPs: {flops}, Number of Parameters: {params}')
    logger.log(f'Model Architecture:\n{model}')
    logger.log(f'Loss Function: {loss_fn}')
    '''Start inference'''

    start_time = time.time()
    tsloss, ts_score_val, ts_label_val = test(
        device, test_dataloader, model, loss_fn)
    np.save(os.path.join(work_dir, 'scores.npy'), np.array(ts_score_val))
    np.save(os.path.join(work_dir, 'True_Value.npy'), np.array(ts_label_val))
    ts_eva_metrics, ts_target = evaluation(
        ts_score_val, ts_label_val, configs['evaluation'], None, None)
    epoch_time = time.time() - start_time
    time_info = dict(time=epoch_time)
    mem_info = AirFunc.get_mem_info(device)
    loss_info = dict(mode='test', test_loss=tsloss)
    log_info = {**loss_info, **time_info, **mem_info}
    json_logger.log(log_info)
    logger.log(AirFunc.readable_dict(log_info, indent='', sep=','))
    ts_eva_info = dict(mode='ts_eval')
    ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
    json_logger.log(ts_eva_info)
    logger.log(AirFunc.readable_dict(ts_eva_info))
    return 0


def evaluation(scores, labels, evaluation_command, best_target, loss):
    metrics = evaluation_command['metrics']
    mode = evaluation_command['sota_target']['mode']
    target = evaluation_command['sota_target']['target']
    eva_res = {}
    for key in metrics:
        eva_res[key] = metrics[key](scores, labels)
    if target is not None and best_target is not None:
        if mode == 'min':
            if eva_res[target] < best_target:
                best_target = eva_res[target]
        elif mode == 'max':
            if eva_res[target] > best_target:
                best_target = eva_res[target]
        return eva_res, best_target
    elif target is not None and best_target is None:
        return eva_res, eva_res[target]
    elif target is None and best_target is not None:
        if loss < best_target:
            return eva_res, loss
        else:
            return eva_res, best_target
    elif target is None and best_target is None:
        return eva_res, loss


def test(device, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch
            y = y.reshape(-1)
            # if num_frames is not None:
            #     y = np.repeat(y, num_frames)
            if isinstance(x, list):
                x = [torch.autograd.Variable(x_).cuda(
                    device, non_blocking=True) for x_ in x]
                h0 = model.module.init_hidden(x[0].size(0))
            else:
                x = torch.autograd.Variable(x).cuda(device, non_blocking=True)
                h0 = model.module.init_hidden(x.size(0))
            y = torch.autograd.Variable(y).cuda(device, non_blocking=True)
            pred = model(x, h0)
            predictions.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


@click.command()
@click.option('--config', default='/home/dachuang2022/Yufeng/DeepMuon/config/Hailing/MLP3_3D.py')
@click.option('--msg', default='')
def run(config, msg):
    train_config = Config(configpath=config)
    if train_config.paras['gpu_config']['distributed'] == True:
        warnings.warn(
            'Distributed Training is not supported during model inference')
    main(config_info=train_config, msg=msg)


if __name__ == '__main__':
    print(f'\n---Starting Neural Network...PID:{os.getpid()}---')
    run()
