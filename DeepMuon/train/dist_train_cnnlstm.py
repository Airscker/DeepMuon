'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-01-31 17:45:58
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
    local_world_size = os.environ['LOCAL_WORLD_SIZE']
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
        log = os.path.join(work_dir, log)
        '''show hyperparameters'''
        logger.log(
            f'========= Current Time: {time.ctime()} Current PID: {os.getpid()} =========')
        logger.log(f'LOCAL WORLD SIZE: {local_world_size}')
        if not os.path.exists(msg):
            logger.log('LICENSE MISSED! REFUSE TO START TRAINING')
            return 0
        with open(msg, 'r') as f:
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
        **configs['model']['params'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
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
    optimizer = configs['optimizer']['backbone'](model.parameters(),
                                                 **configs['optimizer']['params'])
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
    '''Training Initailization'''
    tsloss, ts_score, ts_real = test(device, test_dataloader, model, loss_fn)
    bestres = torch.tensor([tsloss, ts_score, ts_real], device=device)
    # ----------------------------------------------------------
    # Add evaluation pipeline here
    # ----------------------------------------------------------
    dist.barrier()
    dist.all_reduce(bestres)
    bestacc = bestres[1].item()/bestres[2].item()
    if local_rank == 0:
        bar = tqdm(range(epoch_now, epochs), mininterval=1)
    else:
        bar = range(epoch_now, epochs)
    '''Start training'''
    for t in bar:
        start_time = time.time()
        train_dataloader.sampler.set_epoch(t)
        trloss, tr_correct, tr_total = train(
            device, train_dataloader, model, loss_fn, optimizer, scheduler)
        tsloss, ts_correct, ts_total, _ = test(
            device, test_dataloader, model, loss_fn, configs['evaluation']['metrics'])
        '''Synchronize all threads'''
        res = torch.tensor([trloss, tr_correct, tr_total,
                           tsloss, ts_correct, ts_total], device=device)
        dist.barrier()
        '''Reduces the tensor data across all machines in such a way that all get the final result.(Add results of every gpu)'''
        dist.all_reduce(res, op=torch.distributed.ReduceOp.SUM)
        # res = res/float(local_world_size)
        if local_rank == 0:
            train_loss = res[0].item()/local_world_size
            test_loss = res[3].item()/local_world_size
            train_acc = res[1].item()/res[2].item()
            test_acc = res[4].item()/res[5].item()
            LRn = optimizer.state_dict()['param_groups'][0]['lr']
            bar.set_description(
                f'LR: {LRn},Test Loss: {test_loss},Train Loss: {train_loss},Train Top1ACC: {train_acc},Test Top1ACC:{test_acc}')
            writer.add_scalar(f'Test Loss Curve', test_loss, global_step=t + 1)
            writer.add_scalar(f'Train Loss Curve',
                              train_loss, global_step=t + 1)
            writer.add_scalar(f'Test Top1ACC Curve',
                              test_acc, global_step=t + 1)
            writer.add_scalar(f'Train Top1ACC Curve',
                              train_acc, global_step=t + 1)
            if test_acc <= bestacc:
                bestacc = test_acc
                '''Double save to make sure secure, directly save total model is forbidden, otherwise load issues occur'''
                savepath = os.path.join(work_dir, 'Best_Performance.pth')
                AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                                   loss_fn=loss_fn, scheduler=scheduler, path=savepath, dist_train=True)
                AirFunc.save_model(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler, path=os.path.join(
                    work_dir, f'{model_name}_Best_Performance.pth'), dist_train=True)
                logger.log(
                    f'Best Model Saved as {savepath}, Best Test Top1ACC: {bestacc}, Current Epoch: {(t+1)}', show=False)
            if (t + 1) % inter == 0:
                savepath = os.path.join(work_dir, f'Epoch_{t+1}.pth')
                AirFunc.save_model(epoch=t, model=model, optimizer=optimizer,
                                   loss_fn=loss_fn, scheduler=scheduler, path=savepath, dist_train=True)
                logger.log(
                    f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=False)
            epoch_time = time.time() - start_time
            eta = AirFunc.format_time(
                (epochs - 1 - t) * (time.time() - start_time))
            mem_info = get_mem_info()
            logger.log(f"LR: {LRn}, Epoch: [{t+1}][{epochs}], Test Loss: {test_loss}, Train Loss: {train_loss}, Best Test Top1ACC: {bestacc}, Train Top1ACC:{train_acc} Test Top1ACC:{test_acc}, Time:{epoch_time}s, ETA: {eta}, Memory Left: {mem_info['mem_left']} Memory Used: {mem_info['mem_used']}", show=False)
            log_info = dict(mode='train', lr=LRn, epoch=t + 1, total_epoch=epochs, test_loss=test_loss, train_loss=train_loss, test_t1acc=test_acc, train_t1acc=train_acc,
                            best_test_t1acc=bestacc, time=epoch_time, eta=eta, memory_left=mem_info['mem_left'], memory_used=mem_info['mem_used'])
            json_logger.log(log_info)
    return bestacc


def get_mem_info():
    gpu_id = torch.cuda.current_device()
    mem_total = torch.cuda.get_device_properties(gpu_id).total_memory
    mem_cached = torch.cuda.memory_reserved(gpu_id)
    mem_allocated = torch.cuda.memory_allocated(gpu_id)
    return dict(mem_left=f"{(mem_total-mem_cached-mem_allocated)/1024**2:0.2f} MB",
                total_mem=f"{mem_total/1024**2:0.2f} MB",
                mem_used=f"{(mem_cached+mem_allocated)/1024**2:0.2f} MB")


def train(device, dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    train_loss, correct, total = 0.0, 0.0, 0.0
    for i, batch in enumerate(dataloader):
        x, y = batch
        x = x.type(torch.FloatTensor)
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
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += y.size(0)
        predicted = torch.argmax(pred.data, 1)
        correct += predicted.eq(y.data).cpu().sum()
    scheduler.step()
    return train_loss, correct, total


def test(device, dataloader, model, loss_fn):
    model.eval()
    test_correct, test_total, test_loss = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch
            x = x.type(torch.FloatTensor)
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
            outputs = model(x, h0)
            test_loss += loss_fn(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += predicted.eq(y.data).cpu().sum()
            test_total += y.size(0)
    return test_loss, outputs.detach().cpu().numpy(), y.detach().cpu().numpy()


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
