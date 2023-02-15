'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2023-02-15 16:57:11
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''
## Specify which model to be used, all models are stored in 'models' 
'''
model = dict(backbone='ResMax')
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(backbone='HailingDataset_Direct2', params=dict(
    datapath='/data/Airscker/VST3/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma227_tr70k.pkl', augment=False))
test_dataset = dict(backbone='HailingDataset_Direct2', params=dict(
    datapath='/data/Airscker/VST3/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma227_ts10k.pkl', augment=False))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/ResMax_01', logfile='log.log')
'''
## Specify the checkpoint configuration
'''
# checkpoint_config=dict(load_from='',resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/CSPP_3/Best_Performance.pth',save_inter=500)
# checkpoint_config=dict(load_from='',resume_from='',save_inter=500)
checkpoint_config = dict(load_from='', resume_from='', save_inter=500)

'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='MSALoss')
'''
## Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=1000, batch_size=11000, inputshape=[1, 3, 10, 10, 40])
'''
optimizer
'''
optimizer = dict(backbone='AdamW', params=dict(
    lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999)))
'''
scheduler
'''
scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10))
'''
## Specify the GPU config and DDP
'''
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
