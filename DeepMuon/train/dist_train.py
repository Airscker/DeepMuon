'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-10-07 16:00:57
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import time
import os
import click
import numpy as np
import functools
import pickle as pkl
from typing import Union
from multiprocessing.managers import SharedMemoryManager

import DeepMuon
from DeepMuon.tools import (Config,LOGT,EnvINFO,TaskFIFOQueueThread,TaskFIFOQueueProcess,SharedMemory,
                            save_model,load_model,format_time,get_mem_info,
                            load_json_log,generate_nnhs_config,plot_curve)
from DeepMuon.interpret import GradCAM

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

    # '''Using spawn instead of fork to avoid deadlocks in TaskFIFOQueueProcess/Thread'''
    # torch.multiprocessing.set_start_method('spawn')
    '''Initialize the basic training configuration'''
    configs = config_info.paras
    batch_size = configs['hyperpara']['batch_size']
    epochs = configs['hyperpara']['epochs']
    train_params = configs['train_dataset']['params']
    train_num_workers = configs['train_dataset']['num_workers']
    test_params = configs['test_dataset']['params']
    test_num_workers = configs['test_dataset']['num_workers']
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
    try:
        group = dist.init_process_group(backend="nccl")
    except:
        group = dist.init_process_group(backend="gloo")
    local_rank = dist.get_rank()
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    '''Log the basic parameters'''
    if local_rank == 0:
        # '''Initialize (un)sequenced model saving quene'''
        # best_model_save_quene=TaskFIFOQueueProcess(sequenced=False,verbose=True,daemon=True)
        # checkpoint_save_quene=TaskFIFOQueueProcess(sequenced=True,verbose=True,daemon=True)
        # best_model_save_quene.start()
        # checkpoint_save_quene.start()
        logger = LOGT(log_dir=work_dir, logfile=log)
        '''Create work_dir'''
        checkpoint_savepath=os.path.join(work_dir,'Checkpoint')
        curve_path=os.path.join(work_dir,'Figure')
        folders_toCreate=[work_dir,checkpoint_savepath,curve_path]
        for folder in folders_toCreate:
            if not os.path.exists(folder):
                os.makedirs(folder)
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
        env_info=EnvINFO()
        logger.log(env_info)
        logger.log(f'Configuration:\n{config_info}')
        if test_path is not None:
            logger.log(
                f"Test mode enabled, model will be tested upon checkpoint {test_path}")

    '''
    Load datasets
    eg. train_dataset=PandaxDataset(IMG_XY_path=train_data)
        test_dataset=PandaxDataset(IMG_XY_path=test_params)
    In the example shown above, `configs['train_dataset']['backbone']` <> `PandaxDataset`, `IMG_XY_path=train_data` <> `**train_data`
    '''
    _test_load_time=time.time()
    test_dataset = configs['test_dataset']['backbone'](**test_params)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 num_workers=test_num_workers,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 sampler=test_sampler,
                                 collate_fn=configs['test_dataset']['collate_fn'])
    _test_load_time=time.time()-_test_load_time
    if test_path is None:
        _train_load_time=time.time()
        train_dataset = configs['train_dataset']['backbone'](**train_params)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      num_workers=train_num_workers,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=configs['train_dataset']['collate_fn'])
        _train_load_time=time.time()-_train_load_time
    else:
        _train_load_time=0

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
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = load_model(
            path=resume, device=device)
        try:
            model.load_state_dict(model_c, False)
        except:
            pass
        epoch_now = epoch_c + 1
        if local_rank == 0:
            logger.log(f'Model Resumed from {resume}, Epoch now: {epoch_now}')
    elif load != '':
        epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = load_model(
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
        logger.log(f'Loading testing datasets costs {_test_load_time:.4f}s')
        logger.log(f'Loading training datasets costs {_train_load_time:.4f}s')
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
            local_rank], output_device=local_rank, find_unused_parameters=configs['optimize_config']['find_unused_parameters'])
        ddp_training = True
        if not fsdp_env and local_rank == 0:
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
        if local_rank == 0:
            np.save(os.path.join(work_dir,'scores.npy'),ts_score_val)
            np.save(os.path.join(work_dir,'labels.npy'),ts_label_val)
            ts_eva_metrics, _, vis_com= evaluation(
                ts_score_val, ts_label_val, configs['evaluation'], 0, 0,logger)
            for key in ts_eva_metrics.keys():
                if vis_com[key][4] is not None:
                    vis_com[key][4](ts_score_val,ts_label_val,save_path=curve_path)
            ts_eva_info = dict(mode='ts_eval')
            ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
            logger.log(ts_eva_info,json_log=True)
        return 0

    '''Start training, only if test_path==None'''
    bestres = None
    eva_interval = configs['evaluation']['interval']
    best_checkpoint = ''
    sota_target = configs['evaluation']['sota_target']['target']
    if sota_target is None:
        sota_target = 'loss'
    sys_vis_com=dict(loss=('Loss',True,True,True,None),
                     train_loss=('TRLoss',True,True,True,None),
                     test_loss=('TSLoss',True,True,True,None),
                     lr=('Learning Rate',True,True,True,None),
                     sota=('SOTA',True,True,True,None),
                     time=('Time',True,True,True,None),)
    ts_eva_metrics={}
    tr_eva_metrics={}
    for t in range(epoch_now, epochs):
        start_time = time.time()
        train_dataloader.sampler.set_epoch(t)
        trloss, tr_score, tr_label = train(device=device, dataloader=train_dataloader, model_pipeline=model_pipeline, loss_fn=loss_fn, optimizer=optimizer,
                                           scheduler=scheduler, gradient_accumulation=grad_acc, grad_clip=grad_clip, fp16=fp16, grad_scalar=grad_scalar)
        tsloss, ts_score, ts_label = test(device=device, dataloader=test_dataloader, model_pipeline=model_pipeline, loss_fn=loss_fn)

        '''Synchronize all threads, reduces the tensor data across all machines in such a way that all get the final result (Add/Gather results of every gpu).'''
        loss_values = torch.tensor([trloss, tsloss], device=device)
        dist.barrier()
        dist.all_reduce(loss_values, op=torch.distributed.ReduceOp.SUM)

        '''Gather the score and label of the model every `eva_interval` epochs, to be prepared for evaluation.'''
        if (t+1) % eva_interval == 0 and sota_target != 'loss':
            tr_score_val, tr_label_val = gather_score_label(
                tr_score, tr_label, local_world_size)
            ts_score_val, ts_label_val = gather_score_label(
                ts_score, ts_label, local_world_size)
        else:
            tr_score_val, tr_label_val = None, None
            ts_score_val, ts_label_val = None, None
        loss_values = loss_values/float(local_world_size)

        '''Evaluation and save the model'''
        if local_rank == 0:
            train_loss = loss_values[0].item()
            test_loss = loss_values[1].item()
            LRn = optimizer.state_dict()['param_groups'][0]['lr']
            if (t+1)%eva_interval==0:
                ts_eva_metrics, ts_target, vis_com = evaluation(
                    ts_score_val, ts_label_val, configs['evaluation'], bestres, test_loss,False)
                tr_eva_metrics, tr_target, vis_com = evaluation(
                    tr_score_val, tr_label_val, configs['evaluation'], bestres, 0,False)
            else:
                ts_eva_metrics={}
                tr_eva_metrics={}

            '''Visualize records of loss and learning rate as well as evaluation metrics'''
            vis_com={**sys_vis_com,**vis_com}
            writer.add_scalar('Learning rate', LRn, global_step=t + 1)
            tensorboard_plot(metrics={**tr_eva_metrics,'loss':train_loss}, epoch=t+1, writer=writer, tag='TR', vis_com=vis_com)
            tensorboard_plot(metrics={**ts_eva_metrics,'loss':test_loss}, epoch=t+1, writer=writer, tag='TS', vis_com=vis_com)
            if t+1==epochs:
                end_exp=True
            else:
                end_exp=False
            nnhs_report(search=search,
                        sota_target=sota_target,
                        eva_metrics=[{**tr_eva_metrics,'loss':train_loss},{**ts_eva_metrics,'loss':test_loss}],
                        modes=['tr_eval','ts_eval'],
                        end_exp=end_exp,
                        vis_com=vis_com)

            '''Save best model according to the value of sota target'''
            if ts_target != bestres and not np.isnan(ts_target):
                bestres = ts_target
                if os.path.exists(best_checkpoint):
                    os.remove(best_checkpoint)
                best_checkpoint = os.path.join(
                    work_dir, f"Best_{sota_target}_epoch_{t+1}.pth")
                # dist.barrier()
                # best_model_save_quene.add_task(ddp_fsdp_model_save,(t,model,optimizer,loss_fn,scheduler,best_checkpoint,ddp_training),t)
                ddp_fsdp_model_save(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                                    scheduler=scheduler, path=best_checkpoint, ddp_training=ddp_training)
                logger.log(
                    f'Best Model Saved as {best_checkpoint}, Best {sota_target}: {bestres}, Current Epoch: {t+1}', show=True)
            if (t + 1) % inter == 0:
                savepath = os.path.join(checkpoint_savepath,f'Epoch_{t+1}.pth')
                # dist.barrier()
                # checkpoint_save_quene.add_task(ddp_fsdp_model_save,(t,model,optimizer,loss_fn,scheduler,savepath,ddp_training),t)
                ddp_fsdp_model_save(epoch=t, model=model, optimizer=optimizer, loss_fn=loss_fn,
                                    scheduler=scheduler, path=savepath, ddp_training=ddp_training)
                logger.log(
                    f'CheckPoint at epoch {(t+1)} saved as {savepath}', show=True)
            
            '''Record the training information in the log file'''
            epoch_time = time.time() - start_time
            eta = format_time((epochs - 1 - t) * epoch_time)
            time_info = dict(time=epoch_time, eta=eta)
            mem_info = get_mem_info()
            loss_info = dict(mode='train', lr=LRn, epoch=t+1, total_epoch=epochs,
                             test_loss=test_loss, train_loss=train_loss, sota=bestres,
                             batch_size=batch_size, train_dataset_size=len(train_dataset),test_dataset_size=len(test_dataset))
            log_info = {**loss_info, **time_info, **mem_info}
            logger.log(log_info,json_log=True)
            if (t+1) % eva_interval == 0 and tr_eva_metrics != {}:
                tr_eva_info = dict(mode='tr_eval')
                ts_eva_info = dict(mode='ts_eval')
                tr_eva_info = {**tr_eva_info, **tr_eva_metrics}
                ts_eva_info = {**ts_eva_info, **ts_eva_metrics}
                logger.log(tr_eva_info,json_log=True)
                logger.log(ts_eva_info,json_log=True)
    if local_rank==0:
        print('Plotting training information...')
        json_log=load_json_log(logger.jsonfile)
        if json_log=={}:
            return 0
        if not os.path.exists(curve_path):
            os.makedirs(curve_path)
        for mode in json_log.keys():
            for para in json_log[mode].keys():
                if para in vis_com.keys() and vis_com[para][3]:
                    Name=vis_com[para][0]
                    print(f'Plotting records for: {mode}|{para}')
                    plot_curve(data=json_log[mode][para],
                               title=f'{mode}_{Name}',
                               axis_label=['Epoch',f'{Name}'],
                               data_label=[f'{Name}'],
                               save=os.path.join(curve_path,f'{mode}_{para}.jpg'),
                               mod=None)
        # print('Finishing model saving quene...')
        # best_model_save_quene.end_task()
        # checkpoint_save_quene.end_task()
        print('Training finished!')
    return 0

def nnhs_report(search:bool,sota_target:str,eva_metrics:list,modes:list,end_exp:bool,vis_com:dict):
    if search:
        new_metric={}
        for i in range(len(eva_metrics)):
            for key in eva_metrics[i].keys():
                if vis_com[key][1]==False:
                    continue
                else:
                    if key == sota_target and modes[i] == 'ts_eval':
                        new_key='default'
                        # print('SOTA TARGET',modes[i],key,sota_target)
                    else:
                        new_key=f'{modes[i]}_{vis_com[key][0]}'
                    new_metric[new_key]=eva_metrics[i][key]
        if end_exp:
            nni.report_final_result(new_metric)
        else:
            nni.report_intermediate_result(new_metric)

def tensorboard_plot(metrics: dict, epoch: int, writer:SummaryWriter, tag:str, vis_com: dict):
    for key in metrics:
        if vis_com[key][2]:
            writer.add_scalar(f'{tag}_{vis_com[key][0]}', metrics[key], global_step=epoch)

def ddp_fsdp_model_save(epoch=0, model=None, optimizer=None,
                        loss_fn=None, scheduler=None, path=None, ddp_training=True):
    if ddp_training:
        model_data=model.module.state_dict()
        model_bytes=pkl.dumps(model_data)
        shared_mem=SharedMemory(name='MAIN00X1',create=True,size=len(model_bytes))
        shared_mem.buf[:]=model_bytes
        print('TIME SLEEPING FOR 2s')
        time.sleep(10)
        shared_mem.close()
        shared_mem.unlink()
        save_model(epoch=epoch, model=model, optimizer=optimizer,
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


def dataattr(device, dataloader, model):
    attr = []
    delta = []
    convergence = []
    batchs = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        res1, res2, res3 = GradCAM(
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


def evaluation(scores, labels, evaluation_command, best_target, loss, logger):
    '''
    ## Evaluate the model performance

    ### Args:
        - scores: The output of the model.
        - labels: The ground truth of the data.
        - evaluation_command: The evaluation command, which is a dict parsed by `DeepMuon.tools.AirConfig.Config`.
        - best_target: The last best target of the model, if the model can't be better than the former one, the `best_target` will be returned.
        - loss: The loss of the model, if the target of model didn't specified in configuration, then loss value will be the target of model 
            (That is, `best_target` is actually the last smallest loss value).
        - logger: The logger of the training process, which is an instance of `DeepMuon.tools.AirLogger.LOGT`.

    ### Returns:
        - eva_res: The evaluation result of the model, which is a dict contains all the metrics specified in `evaluation_command`.
        - best_target: The best target of the model up to now, which is the best value of the target metric.
        - visual_commands: The commands of visualization, which is a dict indicates the authorizations of visualization of each metric and their visualization methods.

    ### Tips:
        After version 1.23.91, the metric-evaluating methods stored in module `evaluation.py` may return a tuple 
        which contains the metric value as well as the parameters of Visualization Register, 
        more details please refer to `DeepMuon.tools.AirDecorators.EnableVisualiaztion`.
    '''
    metrics = evaluation_command['metrics']
    mode = evaluation_command['sota_target']['mode']
    target = evaluation_command['sota_target']['target']
    eva_res = {}
    visual_commands={}
    '''Get metrics'''
    for key in metrics:
        try:
            eva_res[key] = metrics[key](scores, labels)
            results=metrics[key](scores, labels)
            if isinstance(results,tuple) and results[-1]=='VisualizationRegistered':
                '''result,Name,NNHSReport,TRTensorBoard,TRCurve,TSPlotMethod'''
                eva_res[key]=results[0]
                visual_commands[key]=results[1:-1]
            else:
                eva_res[key]=results
                visual_commands[key]=(key,False,False,False,None)
                logger.log(f"WARNING: After version 1.23.91, the metric-evaluating methods stored in module `evaluation.py` should be decorated by \
                           VisualiaztionRegister `DeepMuon.tools.AirDecorators.EnableVisualiaztion` to be properly visualized.\n\
                           Otherwise some evaluation metrics may occur fetal errors when using default plotting methods.\n\
                           Unless evaluation method `{metrics[key].__name__} is registered for visualization, no record will be visualized for this metric.`".replace('  ',''))
        except:
            pass
    '''
    Four situations:
        - The model has a target metric and model has been trained for at least one epoch.
        - The model has a target metric but training has just started.
        - The loss value is actually the target and model has been trained for at least one epoch.
        - The loss value is actually the target but training has just started.
    '''
    if target is not None and best_target is not None:
        if mode == 'min':
            if eva_res[target] < best_target:
                best_target = eva_res[target]
        elif mode == 'max':
            if eva_res[target] > best_target:
                best_target = eva_res[target]
        return eva_res, best_target,visual_commands
    elif target is not None and best_target is None:
        return eva_res, eva_res[target],visual_commands
    elif target is None and best_target is not None:
        if loss < best_target:
            return eva_res, loss,visual_commands
        else:
            return eva_res, best_target,visual_commands
    elif target is None and best_target is None:
        return eva_res, loss,visual_commands

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
    scheduler.step(train_loss/batchs)
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
        config_module,source_code=generate_nnhs_config(path=config,new_params=new_para)
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
