'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-05-12 00:12:35
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import time
import os
import click
import numpy as np
import functools
from typing import Union

import DeepMuon
from DeepMuon.tools.AirConfig import Config
import DeepMuon.tools.AirFunc as AirFunc
import DeepMuon.tools.AirLogger as AirLogger
import DeepMuon.interpret.attribution as Attr

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
try:
    import nni
    NNHS_enabled=True
except:
    NNHS_enabled=False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    fsdp_env = True
except:
    fsdp_env = False

pkg_path = DeepMuon.__path__[0]
msg = os.path.join(pkg_path.split('DeepMuon')[0], 'LICENSE.txt')
precision=torch.FloatTensor

def main(config_info:Config, test_path:str=None, search:bool=False, source_code:str=None):
    global msg
    global fsdp_env
    global precision
    '''Initialize the basic training configuration'''
    configs = config_info.paras
    batch_size = configs['hyperpara']['batch_size']
    epochs = configs['hyperpara']['epochs']
    train_data = configs['train_dataset']['params']
    test_data = configs['test_dataset']['params']
    work_dir = configs['work_config']['work_dir']
    if search:
        trail_id=nni.get_trial_id()
        work_dir=os.path.join(work_dir,F'NNHS_{nni.get_experiment_id()}',trail_id)
    log = configs['work_config']['logfile']
    if test_path is not None:
        log = 'test_'+log
    resume = configs['checkpoint_config']['resume_from']
    load = configs['checkpoint_config']['load_from']
    fp16 = configs['optimize_config']['fp16']
    grad_scalar = GradScaler(enabled=fp16)
    grad_clip = configs['optimize_config']['grad_clip']
    grad_acc = configs['optimize_config']['grad_acc']
    if configs['optimize_config']['double_precision']:
        precision=torch.DoubleTensor
    torch.set_default_tensor_type(precision)
    if test_path is not None:
        load = test_path
        resume = ''
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
        if search:
            config_info.move_config(source_code=source_code,save_path=os.path.join(work_dir,'config.py'))
        else:
            config_info.move_config()
        log = os.path.join(work_dir, log)
        '''show hyperparameters'''
        logger.log(
            f'========= Current Time: {time.ctime()} Current PID: {os.getpid()} =========')
        logger.log(f'LOCAL WORLD SIZE: {local_world_size}')
        logger.log(f"PORT: {os.environ['MASTER_PORT']}")
        if search:
            logger.log('Neural network hyperparameter searching enabled')
            logger.log(f'NNHS expeirment ID: {nni.get_experiment_id()}')
        if not os.path.exists(msg):
            logger.log('LICENSE MISSED! REFUSE TO START TRAINING')
            return 0
        with open(msg, 'r')as f:
            msg = f.read()
        logger.log(msg)
        logger.log(config_info)
        if test_path is not None:
            logger.log(
                f"Test mode enabled, model will be tested upon checkpoint {test_path}")
    '''
    Load datasets
    eg. train_dataset=PandaxDataset(IMG_XY_path=train_data)
        test_dataset=PandaxDataset(IMG_XY_path=test_data)
    In the example shown above, `configs['train_dataset']['backbone']` <> `PandaxDataset`, `IMG_XY_path=train_data` <> `**train_data`
    '''
    test_dataset = configs['test_dataset']['backbone'](**test_data)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=test_sampler)
    if test_path is None:
        train_dataset = configs['train_dataset']['backbone'](**train_data)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
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
        try:
            model.load_state_dict(model_c, False)
        except:
            pass
        epoch_now = epoch_c + 1
        if local_rank == 0:
            logger.log(f'Model Resumed from {resume}, Epoch now: {epoch_now}')
    elif load != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
            path=load, device=device)
        try:
            model.load_state_dict(model_c, False)
        except:
            pass
        epoch_now = 0
        if local_rank == 0:
            logger.log(
                f'Pretrained Model Loaded from {load}, Epoch now: {epoch_now}')
    epochs += epoch_now
    '''save model architecture before model parallel'''
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(work_dir, 'LOG'))
        # writer.add_graph(model, torch.randn(
        #     configs['hyperpara']['inputshape']).to(device))
    '''Model Parallel'''
    if configs['fsdp_parallel']['enabled'] and fsdp_env:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=configs['fsdp_parallel']['min_num_params']
        )
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
        ddp_training = False
    else:
        model = DistributedDataParallel(model, device_ids=[
            local_rank], output_device=local_rank, find_unused_parameters=False)
        ddp_training = True
        if fsdp_env and local_rank == 0:
            logger.log(
                f'WARN: FSDP is not supported at current edition of torch: {torch.__version__}, we have switched to DDP to avoid mistakes')
    '''
    Initialize loss/optimizer/scheduler
    eg. loss_fn=nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0)
    In the example shown above:
        `nn.MSELoss` <> `configs['loss_fn']['backbone']`, `loss_function parameters` <> `**configs['loss_fn']['params']`
        `torch.optim.SGD` <> `configs['optimizer']['backbone']`, `lr=0.001, momentum=0.9, nesterov=True` <> `**configs['optimizer']['params']`
        `torch.optim.lr_scheduler.ReduceLROnPlateau` <> `configs['scheduler']['backbone']`, `mode='min', factor=0.5, patience=100` <> `**configs['scheduler']['params']`
    '''
    loss_fn = configs['loss_fn']['backbone'](**configs['loss_fn']['params']).to(device)
    if test_path is None:
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
        if test_path is None:
            logger.log(f'Optimizer:\n{optimizer}')
            logger.log(
                f'Scheduler: \n\t{scheduler.__class__.__name__}:\n\t{scheduler.state_dict()}')
    '''Start testing, only if test_path!=None'''
    model_pipeline=configs['model']['pipeline'](model)
    if test_path is not None:
        _, ts_score, ts_label=test(device=device, dataloader=test_dataloader, model_pipeline=model_pipeline, loss_fn=loss_fn)
        # attr,_,_=dataattr(device,test_dataloader,model)
        dist.barrier()
        # all_attr=gather_attr(attr,local_world_size)
        # np.save(os.path.join(work_dir,'attr.npy'),all_attr)
        ts_score_val, ts_label_val = gather_score_label(
            ts_score, ts_label, local_world_size)
        ts_eva_metrics, _ = evaluation(
            ts_score_val, ts_label_val, configs['evaluation'], 0, 0)
        ts_eva_info = dict(mode='ts_eval')
        ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
        if local_rank == 0:
            logger.log(AirFunc.readable_dict(ts_eva_info))
            json_logger.log(ts_eva_info)
        return 0
    '''Start training, only if test_path==None'''
    bestres = None
    eva_interval = configs['evaluation']['interval']
    best_checkpoint = ''
    sota_target = configs['evaluation']['sota_target']['target']
    if sota_target is None:
        sota_target = 'loss'
    for t in range(epoch_now, epochs):
        start_time = time.time()
        train_dataloader.sampler.set_epoch(t)
        trloss, tr_score, tr_label = train(device=device, dataloader=train_dataloader, model_pipeline=model_pipeline, loss_fn=loss_fn, optimizer=optimizer,
                                           scheduler=scheduler, gradient_accumulation=grad_acc, grad_clip=grad_clip, fp16=fp16, grad_scalar=grad_scalar)
        tsloss, ts_score, ts_label = test(device=device, dataloader=test_dataloader, model_pipeline=model_pipeline, loss_fn=loss_fn)
        '''
        Synchronize all threads
        Reduces the tensor data across all machines in such a way that all get the final result.(Add/Gather results of every gpu)
        '''
        loss_values = torch.tensor([trloss, tsloss], device=device)
        dist.barrier()
        dist.all_reduce(loss_values, op=torch.distributed.ReduceOp.SUM)
        if (t+1) % eva_interval == 0 and sota_target != 'loss':
            tr_score_val, tr_label_val = gather_score_label(
                tr_score, tr_label, local_world_size)
            ts_score_val, ts_label_val = gather_score_label(
                ts_score, ts_label, local_world_size)
        else:
            tr_score_val, tr_label_val = None, None
            ts_score_val, ts_label_val = None, None
        loss_values = loss_values/float(local_world_size)
        if local_rank == 0:
            train_loss = loss_values[0].item()
            test_loss = loss_values[1].item()
            LRn = optimizer.state_dict()['param_groups'][0]['lr']
            ts_eva_metrics, ts_target = evaluation(
                ts_score_val, ts_label_val, configs['evaluation'], bestres, test_loss)
            tr_eva_metrics, tr_target = evaluation(
                tr_score_val, tr_label_val, configs['evaluation'], bestres, 0)
            '''Add tensorboard scalar curves'''
            writer.add_scalar('test loss', test_loss, global_step=t + 1)
            writer.add_scalar('train loss', train_loss, global_step=t + 1)
            writer.add_scalar('learning rate', LRn, global_step=t + 1)
            if (t+1) % eva_interval == 0 and sota_target != 'loss':
                tensorboard_plot(tr_eva_metrics, t+1, writer, 'train')
                tensorboard_plot(ts_eva_metrics, t+1, writer, 'test')
            '''Save best model accoeding to the value of sota target'''
            if ts_target != bestres and not np.isnan(ts_target):
                bestres = ts_target
                if os.path.exists(best_checkpoint):
                    os.remove(best_checkpoint)
                best_checkpoint = os.path.join(
                    work_dir, f"Best_{sota_target}_epoch_{t+1}.pth")
                # dist.barrier()
                ddp_fsdp_model_save(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                                    scheduler=scheduler, path=best_checkpoint, ddp_training=ddp_training)
                logger.log(
                    f'Best Model Saved as {best_checkpoint},Best {sota_target}:{bestres}, Current Epoch: {t+1}', show=True)
            if (t + 1) % inter == 0:
                savepath = os.path.join(work_dir, f'Epoch_{t+1}.pth')
                # dist.barrier()
                ddp_fsdp_model_save(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                                    scheduler=scheduler, path=savepath, ddp_training=ddp_training)
                logger.log(
                    f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=True)
            epoch_time = time.time() - start_time
            eta = AirFunc.format_time((epochs - 1 - t) * epoch_time)
            time_info = dict(time=epoch_time, eta=eta)
            mem_info = AirFunc.get_mem_info(device)
            loss_info = dict(mode='train', lr=LRn, epoch=t+1, total_epoch=epochs,
                             test_loss=test_loss, train_loss=train_loss, sota=bestres,
                             batch_size=batch_size, train_dataset_size=len(train_dataset),test_dataset_size=len(test_dataset))
            log_info = {**loss_info, **time_info, **mem_info}
            json_logger.log(log_info)
            logger.log(AirFunc.readable_dict(log_info, indent='', sep=','))
            if (t+1) % eva_interval == 0 and sota_target != 'loss':
                tr_eva_info = dict(mode='tr_eval')
                ts_eva_info = dict(mode='ts_eval')
                tr_eva_info = {**tr_eva_info, **tr_eva_metrics}
                ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
                json_logger.log(tr_eva_info)
                json_logger.log(ts_eva_info)
                logger.log(AirFunc.readable_dict(tr_eva_info))
                logger.log(AirFunc.readable_dict(ts_eva_info))
                if t+1==epochs:
                    end_exp=True
                else:
                    end_exp=False
                nnhs_report(search=search,sota_target=sota_target,eva_metrics=[tr_eva_metrics,ts_eva_metrics],modes=['tr_eval','ts_eval'],end_exp=end_exp)
    return 0

def nnhs_report(search:bool,sota_target:str,eva_metrics:list,modes:list,end_exp:bool):
    if search:
        new_metric={}
        for i in range(len(eva_metrics)):
            for key in eva_metrics[i].keys():
                if key == sota_target and modes[i] == 'ts_eval':
                    new_key='default'
                    print('SOTA TARGET',modes[i],key,sota_target)
                else:
                    new_key=f'{modes[i]}_{key}'
                new_metric[new_key]=eva_metrics[i][key]
        if end_exp:
            nni.report_final_result(new_metric)
        else:
            nni.report_intermediate_result(new_metric)


def ddp_fsdp_model_save(epoch=0, model=None, optimizer=None,
                        loss_fn=None, scheduler=None, path=None, ddp_training=True):
    if ddp_training:
        AirFunc.save_model(epoch=epoch, model=model, optimizer=optimizer,
                           loss_fn=loss_fn, scheduler=scheduler, path=path, dist_train=ddp_training)
    else:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model': cpu_state,
            'optimizer': optimizer.state_dict(),
            'loss_fn': loss_fn.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, path)


def tensorboard_plot(metrics: dict, epoch: int, writer, tag):
    for key in metrics:
        try:
            writer.add_scalar(f'{tag}_{key}', metrics[key], global_step=epoch)
        except:
            pass


def dataattr(device, dataloader, model):
    attr = []
    delta = []
    convergence = []
    batchs = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        res1, res2, res3 = Attr.GradCAM(
            model, model.pre_mlp, x, label_dim=len(y.reshape(-1)), device=device)
        attr.append(res1)
        delta.append(res2)
        convergence.append(res3)
        if device.index == 0:
            print(f'{i}/{batchs} processed')
    return np.concatenate(attr, axis=1), np.concatenate(delta, axis=1), convergence


def gather_attr(data, world_size):
    gathered_data = [None]*world_size
    dist.all_gather_object(gathered_data, data)
    return np.concatenate(gathered_data, axis=1)


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
        try:
            eva_res[key] = metrics[key](scores, labels)
        except:
            pass
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
def release_cache():
    try:
        torch.cuda.empty_cache()
    except:
        pass

def train(device: Union[int, str, torch.device],
          dataloader: DataLoader,
          model_pipeline:DeepMuon.train.pipeline.Pipeline,
          loss_fn=None,
          optimizer=None,
          scheduler=None,
          gradient_accumulation: int = 8,
          grad_clip: float = None,
          fp16: bool = False,
          grad_scalar: GradScaler = None):
    '''
    ## Train model and refrensh its gradients & parameters

    ### Tips:
        - Gradient accumulation: Gradient accumulation steps
        - Mixed precision: Mixed precision training is allowed
        - Gradient resacle: Only available when mixed precision training is enabled, to avoid the gradient exploration/annihilation bring by fp16
        - Gradient clip: Using gradient value clip technique
    '''
    model_pipeline.model.train()
    train_loss = 0
    predictions = []
    labels = []
    batchs = len(dataloader)
    gradient_accumulation = min(batchs, gradient_accumulation)
    for i, (x, y) in enumerate(dataloader):
        with autocast(enabled=fp16):
            if (i+1) % gradient_accumulation != 0:
                with model_pipeline.model.no_sync():
                    pred,label=model_pipeline.predict(input=x,label=y,device=device,precision=precision)
                    loss = loss_fn(pred, label)
                    loss = loss/gradient_accumulation
                    grad_scalar.scale(loss).backward()
            elif (i+1) % gradient_accumulation == 0:
                pred,label=model_pipeline.predict(input=x,label=y,device=device,precision=precision)
                loss = loss_fn(pred, label)
                loss = loss/gradient_accumulation
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        model_pipeline.model.parameters(), grad_clip)
                grad_scalar.scale(loss).backward()
                grad_scalar.step(optimizer)
                grad_scalar.update()
                optimizer.zero_grad()
        predictions.append(pred.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        train_loss += loss.item()*gradient_accumulation
    scheduler.step()
    release_cache()
    return train_loss/batchs, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


def test(device:Union[int, str, torch.device],
         dataloader:DataLoader,
         model_pipeline:DeepMuon.train.pipeline.Pipeline,
         loss_fn=None):
    num_batches = len(dataloader)
    model_pipeline.model.eval()
    test_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            pred,label=model_pipeline.predict(input=x,label=y,device=device,precision=precision)
            predictions.append(pred.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            test_loss += loss_fn(pred, label).item()
    test_loss /= num_batches
    release_cache()
    return test_loss, np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


@click.command()
@click.option('--config', default='', help='Specify the path of configuartion file')
@click.option('--test', default='', help='Specify the path of checkpoint used to test the model performance, if nothing given the test mode will be disabled')
@click.option('--search',is_flag=True,help='Specify whether to use Neural Network Hyperparameter Searching (NNHS for short)')
def start_exp(config, test, search, main_func=main):
    global NNHS_enabled
    if not NNHS_enabled:
        search=False
    source_code=None
    if search:
        new_para=nni.get_next_parameter()
        config_module,source_code=AirFunc.generate_nnhs_config(path=config,new_params=new_para)
        train_config = Config(config_module=config_module)
    else:
        train_config = Config(configpath=config)
    if not os.path.exists(test) and test != '':
        test = None
        print(f"checkpoint {test} cannot be found, test mode is disabled!")
        return 0
    elif test == '':
        test = None
    main_func(train_config, test, search, source_code)

 
if __name__ == '__main__':
    print(f'\n---Starting Neural Network...PID:{os.getpid()}---')
    start_exp()
