'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2023-01-30 21:06:28
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (C) 2022 by Airscker(Yufeng), All Rights Reserved. 
'''
'''
## Specify which model to be used, all models are stored in 'models' 
'''
model = dict(backbone='VST')
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(backbone='HailingDataset_DirectV3', params=dict(
    datapath='/data/Airscker/VST3/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma227_tr70k.pkl', augment=True))
test_dataset = dict(backbone='HailingDataset_DirectV3', params=dict(
    datapath='/data/Airscker/VST3/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma227_ts10k.pkl', augment=False))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/VST_1', logfile='log.log')
'''
## Specify the checkpoint configuration
'''
# checkpoint_config=dict(load_from='',resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/CSPP_3/Best_Performance.pth',save_inter=500)
checkpoint_config = dict(
    load_from='', resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/VST_1/Best_Performance.pth', save_inter=500)
# checkpoint_config = dict(load_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/DResMax_3/Best_Performance.pth', resume_from='', save_inter=500)
'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='MSALoss')
'''
## Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=2000, batch_size=7500, inputshape=[1, 3, 40, 10, 10])
'''
## Specify the lr as well as its config, the lr will be optimized using torch.optim.lr_scheduler.ReduceLROnPlateau()
'''
lr_config = dict(init=0.0001, patience=100)
'''
## Specify the GPU config and DDP
'''
gpu_config = dict(distributed=True, gpuid=0)
