'''
Author: Airscker
Date: 2022-07-19 13:01:17
LastEditors: airscker
LastEditTime: 2023-02-16 16:57:00
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import time
import os
import click
import warnings
import numpy as np
import pickle as pkl

from DeepMuon.tools.AirConfig import Config
import DeepMuon.tools.AirFunc as AirFunc
import DeepMuon.tools.AirLogger as AirLogger
from DeepMuon.tools.model_info import model_para
from DeepMuon.interpret.analysis import loss_dist, data_analysis
from DeepMuon.loss_fn.evaluation import confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from ptflops import get_model_complexity_info
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(profile='full')


def main(config_info, ana, thres, neuron):
    '''Initialize the basic training configuration'''
    configs = config_info.paras
    loss_fn = configs['loss_fn']['backbone'](configs['loss_fn']['params'])
    batch_size = 1
    test_data = configs['test_dataset']['params']
    work_dir = configs['work_config']['work_dir']
    assert os.path.exists(
        work_dir), f'The work_dir specified in the config file can not be found: {work_dir}'
    log = 'inference.log'
    infer_path = os.path.join(work_dir, 'infer')
    res = os.path.join(infer_path, 'inference_res.pkl')
    load = os.path.join(work_dir, 'Best_Performance.pth')
    gpu = configs['gpu_config']['gpuid']
    logger = AirLogger.LOGT(log_dir=work_dir, logfile=log)
    json_logger = AirLogger.LOGJ(log_dir=work_dir, logfile=f'{log}.json')
    log = os.path.join(infer_path, log)
    ana_path = os.path.join(work_dir, 'ana')

    '''load datasets'''
    test_dataset = configs['test_dataset']['backbone'](**test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    '''Get cpu or gpu device for training.'''
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    '''show hyperparameters'''
    logger.log(f'========= Current Time: {time.ctime()} =========')
    logger.log(f'Current PID: {os.getpid()}')
    config_path = configs['config']['path']
    logger.log(f'Configuration loaded from {config_path}')
    logger.log(f'Batch size is set as {batch_size} during model inference')
    logger.log(f'Test Dataset loaded from {test_data}')
    logger.log(f'Device Used: {device}')
    logger.log(f'Work_dir of the model: {work_dir}')
    logger.log(
        f'Inference results will be save into work_dir of the model\nLog info will be saved into {log}')
    logger.log(f'All inference results will be saved into {res}')
    if ana:
        logger.log(f'Data analysis results will be saved into {ana_path}')

    # You can change the name of net as any you want just make sure the model structure is the same one
    model = configs['model']['backbone'](
        **configs['model']['params']).to(device)
    assert os.path.exists(load), f'Model inferenced can not be found: {load}'
    epoch_c, model_c, optimizer_c, scheduler_c, loss_fn_c = AirFunc.load_model(
        path=load, device=device)
    model.load_state_dict(model_c, False)
    model.to(device)
    logger.log(f'Pretrained Model Loaded from {load}')
    model_name = model._get_name()

    # Get GFLOPS of the model
    flops, params, sumres = model_para(
        model=configs['model']['backbone'](**configs['model']['params']), datasize=configs['hyperpara']['inputshape'], depth=10, gpu=gpu)
    logger.log(f'Model Architecture:\n{model}')
    logger.log(f'{sumres}')
    logger.log(
        f'Overall Model GFLOPs: {flops}, Number of Parameters: {params}')
    logger.log(f'Loss Function: {loss_fn}')

    '''save model architecture'''
    # writer=SummaryWriter(os.path.join(work_dir,'LOG'))
    # writer.add_graph(model,torch.rand(configs['hyperpara']['inputshape']).to(device))
    # if neuron >= 0:
    #     neuron_infer(device=device, dataloader=test_dataloader,
    #                  model=model, loss_fn=loss_fn, work_dir=work_dir, index=neuron)
    '''start inferencing'''
    loss, pred, real, loss_map = infer(
        work_dir, device, test_dataloader, model, loss_fn, logger, thres=thres, ana=ana)
    # Save results

    with open(res, 'wb')as f:
        pkl.dump({'loss': loss, 'pred': pred, 'real': real}, f)
    f.close()
    logger.log(
        f'Inference Results Saved in {res}\nAvailable Results: dict_keys(loss pred real)')
    # Analysis the results
    if ana == True:
        if not os.path.exists(ana_path):
            os.makedirs(ana_path)
        # loss=np.array(loss)
        # name=os.path.join(ana_path,'loss_dist.jpg')
        # loss_dist(loss,save=name)
        data_analysis(thres, sigma=3, config=config_path)
        logger.log(
            f'Distribution Histogram Image of Loss value is saved into {ana_path}')
        logger.log(
            f'The loss threshold is set as {thres}\nLoss MEAN STD: {np.mean(loss)} {np.std(loss)}')
        name = os.path.join(ana_path, f'loss_thres_{thres}.pkl')
        with open(name, 'wb')as f:
            pkl.dump(loss_map, f)
        f.close()
        logger.log(
            f'Threshold {thres} Loss ID-[data,pred,real,loss] hash map saved as {name}, total number: {len(loss_map)}')


def infer(workdir, device, dataloader, model, loss_fn, logger: AirLogger.LOGT, thres, ana=True):
    """
    The infer function is used to test the model on a dataset. It takes in a dataloader, 
    a model, and an optional loss function. The loss function is only needed if you want to 
    see the accuracy of your predictions (for example, if you are training). This will return 
    the average loss over all of the batches as well as two lists: one with predicted values for each batch and one with real values for each batch.

    :param device: Specify which device to use
    :param dataloader: Generate the data
    :param model: Specify the model to be used
    :param loss_fn: Calculate the loss of our model
    :param logger:LOGT: Log the training process
    :param thres: Filter out the abnormal data
    :param ana=True: Determine whether to analyze the data
    :return: The loss value, predicted values and real values
    """
    num_batches = len(dataloader)
    model.eval()
    test_loss = []
    pred_value = []
    real_value = []
    num = 1
    loss_map = {}

    with torch.no_grad():
        for X, y in dataloader:
            start_time = time.time()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_value = loss_fn(pred, y).item()
            test_loss.append(loss_value)
            pred_value.append(pred.detach().cpu().numpy())
            real_value.append(y.detach().cpu().numpy())
            if ana and loss_value > thres:
                loss_map[num-1] = [X.cpu().numpy(), pred_value[num-1],
                                   real_value[num-1], loss_value]
            now_time = time.time()
            logger.log(
                f'{num}/{num_batches} Loss: {loss_value}, Predicted: {pred_value[num-1]}, Real: {real_value[num-1]}, Time: {now_time-start_time}s, ETA: {AirFunc.format_time((num_batches-num)*(now_time-start_time))}')
            num += 1
    return test_loss, pred_value, real_value, loss_map


# def neuron_infer(device, dataloader, model: nn.Module, loss_fn, work_dir: str, index=0):
#     """
#     The neuron_infer function is used to infer the neuron values of a given model.
#     It takes in the following parameters:
#         - device: The torch device on which the computations will be run. This is typically set to cuda if you have a GPU available, otherwise it should be set to cpu.
#         - dataloader: A DataLoader object that can load your test dataset batch by batch (for example, ImageFolder from torchvision). It should return pairs of images and labels for each iteration.
#         - model: The PyTorch neural network model that we want to infer neurons for (in this case

#     :param device: Specify the device to use
#     :param dataloader: Load the data
#     :param model:nn.Module: Specify the model to be profiled
#     :param loss_fn: Calculate the loss of the model
#     :param work_dir:str: Specify the directory where the log file and neuron
#     :param index=0: Specify the index of the data in the dataset
#     :return: The loss value, the predicted value and the real value of a data point
#     """
#     num_batches = len(dataloader)
#     model.eval()
#     num = 1
#     neuron_log = os.path.join(work_dir, 'neuron')
#     logger = LOGT(log_dir=neuron_log, logfile='neuron.log', new=True)
#     with torch.no_grad():
#         for X, y in dataloader:
#             if num-1 == index:
#                 start_time = time.time()
#                 X, y = X.to(device), y.to(device)
#                 with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True) as prof_ana:
#                     pred = model(X)
#                 table = prof_ana.table()
#                 logger.log(f'Model architecture: \n{model}\n\n')
#                 logger.log(table, show=False)
#                 prof_ana.export_chrome_trace(
#                     os.path.join(neuron_log, 'model_profile.json'))
#                 logger.log('\nModel Parameters:')
#                 for name, param in model.named_parameters():
#                     logger.log(
#                         f"Layer: {name} | Size: {param.size()} | Values : {param} \n", show=False)

#                 '''Trace Feature Map'''
#                 logger.log(
#                     f'\nFeature map of data with index {index} in the dataset\n', show=False)
#                 nodes, _ = get_graph_node_names(model)
#                 fx = create_feature_extractor(model, return_nodes=nodes)
#                 fms = fx(X)
#                 for key in fms.keys():
#                     logger.log(
#                         f'Feature: {key}\nValue: \n{fms[key]}\n', show=False)
#                 with open(os.path.join(neuron_log, 'FX.pkl'), 'wb')as f:
#                     pkl.dumps(fms)

#                 loss_value = loss_fn(pred, y).item()
#                 test_loss = loss_value
#                 pred_value = pred.cpu().numpy()[0]
#                 real_value = y.cpu().numpy()[0]
#                 now_time = time.time()
#                 logger.log(
#                     f'{index} Loss: {loss_value}, Predicted: {pred_value}, Real: {real_value}, Time: {now_time-start_time}s')
#                 break
#             num += 1
#     return test_loss, pred_value, real_value


@click.command()
@click.option('--config', '-c', default='/home/dachuang2022/Yufeng/DeepMuon/config/Hailing/Vit.py')
@click.option('--neuron', '-n', default=-1)
@click.option('--ana', '-a', default=True)
@click.option('--thres', '-t', default=0.004)
def run(config, neuron, ana, thres):
    """
    The run function is the main function that runs the training and testing of a model.
    It takes in a config file path, an analysis type (either 'train' or 'test'), 
    a threshold value for determining whether to classify as positive or negative, and 
    the neuron number you want to analyze. It then runs the training/testing process using 
    the parameters specified in that config file.

    :param config: Pass the path to the config file
    :param neuron: Specify which data in the dataset to evaluate, if -1 specified the operation will be canceled
    :param ana: Specify which analysis to run
    :param thres: Specify the threshold for the model to be used in inference
    :return: The accuracy of the model
    """
    train_config = Config(configpath=config)
    if train_config.paras['gpu_config']['distributed'] == True:
        warnings.warn(
            'Distributed Training is not supported during model inference')
    main(train_config, ana, thres, neuron)


if __name__ == '__main__':
    print('\n---Starting Neural Network...---')
    run()
