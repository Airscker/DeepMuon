'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-02-04 19:05:14
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import time
import os
import click
import numpy as np

from DeepMuon.tools.AirConfig import Config
import DeepMuon.tools.AirFunc as AirFunc
import DeepMuon.tools.AirLogger as AirLogger

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
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
    '''Initialize Distributed Training'''
    group = torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    '''Log the basic parameters'''
    if local_rank == 0:
        logger = AirLogger.LOGT(log_dir=work_dir, logfile=log)
        json_logger = AirLogger.LOGJ(log_dir=work_dir, logfile=f'{log}.json')
        '''Create work_dir'''
        try:
            os.makedirs(work_dir)
        except:
            pass
        config_info.move_config()
        log = os.path.join(work_dir, log)
        '''show hyperparameters'''
        logger.log(
            f'========= Current Time: {time.ctime()} Current PID: {os.getpid()} =========')
        logger.log(f'LOCAL WORLD SIZE: {local_world_size}')
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
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=test_sampler)
    '''
    Create Model and optimizer/loss/scheduler
    You can change the name of net as any you want just make sure the model structure is the same one
    eg. model = MLP3().to(device)
    In the example shown above, `MLP3` <> `configs['model']['backbone']`, `model_parameters` <> `**configs['model']['params']`
    '''
    model: nn.Module = configs['model']['backbone'](
        **configs['model']['params']).to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    epoch_now = 0
    if resume == '' and load == '':
        pass
    elif resume != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
            path=resume, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = epoch_c + 1
        if local_rank == 0:
            logger.log(f'Model Resumed from {resume}, Epoch now: {epoch_now}')
    elif load != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
            path=load, device=device)
        model.load_state_dict(model_c, False)
        model.to(device)
        epoch_now = 0
        if local_rank == 0:
            logger.log(
                f'Pretrained Model Loaded from {load}, Epoch now: {epoch_now}')
    epochs += epoch_now
    model_name = model._get_name()
    '''save model architecture before model parallel'''
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(work_dir, 'LOG'))
        # writer.add_graph(model, torch.randn(
        #     configs['hyperpara']['inputshape']).to(device))
    '''Model Parallel'''
    model = DistributedDataParallel(model, device_ids=[
                                    local_rank], output_device=local_rank, find_unused_parameters=False)
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
    if local_rank == 0:
        # flops, params = get_model_complexity_info(model,(1,17,17),as_strings=True,print_per_layer_stat=False, verbose=True)
        # logger.log(f'GFLOPs: {flops}, Number of Parameters: {params}')
        logger.log(f'Model Architecture:\n{model}')
        logger.log(f'Loss Function: {loss_fn}')
        logger.log(f'Optimizer:\n{optimizer}')
        logger.log(
            f'Scheduler: {scheduler.__class__.__name__}:\n\t{scheduler.state_dict()}')
    '''Start training'''
    bestres = None
    bar = range(epoch_now, epochs)
    eva_interval = configs['evaluation']['interval']
    for t in bar:
        start_time = time.time()
        train_dataloader.sampler.set_epoch(t)
        trloss, tr_score, tr_label = train(
            device, train_dataloader, model, loss_fn, optimizer, scheduler)
        tsloss, ts_score, ts_label = test(
            device, test_dataloader, model, loss_fn)
        '''
        Synchronize all threads
        Reduces the tensor data across all machines in such a way that all get the final result.(Add/Gather results of every gpu)
        '''
        loss_values = torch.tensor([trloss, tsloss], device=device)
        dist.barrier()
        dist.all_reduce(loss_values, op=torch.distributed.ReduceOp.SUM)
        tr_score_val, tr_label_val = gather_score_label(
            tr_score, tr_label, local_world_size)
        ts_score_val, ts_label_val = gather_score_label(
            ts_score, ts_label, local_world_size)
        loss_values = loss_values/float(local_world_size)
        if local_rank == 0:
            train_loss = loss_values[0].item()/local_world_size
            test_loss = loss_values[1].item()/local_world_size
            LRn = optimizer.state_dict()['param_groups'][0]['lr']
            ts_eva_metrics, ts_target = evaluation(
                ts_score_val, ts_label_val, configs['evaluation'], bestres, tsloss)
            tr_eva_metrics, tr_target = evaluation(
                tr_score_val, tr_label_val, configs['evaluation'], bestres, 0)
            '''Add tensorboard scalar curves'''
            writer.add_scalar('test loss', test_loss, global_step=t + 1)
            writer.add_scalar('train loss', train_loss, global_step=t + 1)
            writer.add_scalar('learning rate', LRn, global_step=t + 1)
            if (t+1) % eva_interval == 0:
                tensorboard_plot(tr_eva_metrics, t+1, writer, 'train')
                tensorboard_plot(ts_eva_metrics, t+1, writer, 'test')
            '''Save best model accoeding to the value of sota target'''
            if ts_target != bestres:
                bestres = ts_target
                sota_target = configs['evaluation']['sota_target']['target']
                if sota_target is None:
                    sota_target = 'loss'
                savepath = os.path.join(
                    work_dir, f"Best_{sota_target}_epoch_{t+1}.pth")
                AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                                   loss_fn=loss_fn, scheduler=scheduler, path=savepath, dist_train=True)
                logger.log(
                    f'Best Model Saved as {savepath},Best {sota_target}:{bestres}, Current Epoch: {t+1}', show=True)
            if (t + 1) % inter == 0:
                savepath = os.path.join(work_dir, f'Epoch_{t+1}.pth')
                AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                                   loss_fn=loss_fn, scheduler=scheduler, path=savepath, dist_train=True)
                logger.log(
                    f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=True)
            epoch_time = time.time() - start_time
            eta = AirFunc.format_time((epochs - 1 - t) * epoch_time)
            time_info = dict(time=epoch_time, eta=eta)
            mem_info = AirFunc.get_mem_info(device)
            loss_info = dict(mode='train', lr=LRn, epoch=t+1, total_epoch=epochs,
                             test_loss=test_loss, train_loss=train_loss, sota=bestres)
            log_info = {**loss_info, **time_info, **mem_info}
            json_logger.log(log_info)
            logger.log(AirFunc.readable_dict(log_info, indent='', sep=','))
            if (t+1) % eva_interval == 0:
                tr_eva_info = dict(mode='tr_eval')
                ts_eva_info = dict(mode='ts_eval')
                tr_eva_info = {**tr_eva_info, **tr_eva_metrics}
                ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
                json_logger.log(ts_eva_info)
                json_logger.log(tr_eva_info)
                logger.log(AirFunc.readable_dict(ts_eva_info))
                logger.log(AirFunc.readable_dict(tr_eva_info))
    return bestres


def tensorboard_plot(metrics: dict, epoch: int, writer, tag):
    for key in metrics:
        try:
            writer.add_scalar(f'{tag}_{key}', metrics[key], global_step=epoch)
        except:
            pass


def gather_score_label(score, label, world_size):
    gathered_data = [None]*world_size
    dist.all_gather_object(gathered_data, [score, label])
    scores = []
    labels = []
    for data in gathered_data:
        scores.append(data[0])
        labels.append(data[1])
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels


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
    elif target is None and best_target is None:
        return eva_res, loss


def train(device, dataloader, model, loss_fn, optimizer, scheduler, gradient_accumulation=8):
    model.train()
    train_loss = 0
    predictions = []
    labels = []
    batchs = len(dataloader)
    gradient_accumulation = min(batchs, gradient_accumulation)
    for i, (x, y) in enumerate(dataloader):
        y = y.reshape(-1).to(device)
        pred = model(x, device)
        predictions.append(pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        loss = loss_fn(pred, y)
        loss = loss/gradient_accumulation
        loss.backward()
        train_loss += loss.item()
        if (i+1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()
    return train_loss/batchs, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


def test(device, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            y = y.reshape(-1).to(device)
            pred = model(x, device)
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
        main(train_config, msg)
    else:
        print('Single GPU Training is not supported!')


if __name__ == '__main__':
    print(f'\n---Starting Neural Network...PID:{os.getpid()}---')
    run()
