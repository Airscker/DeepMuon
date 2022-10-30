'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2022-10-11 20:56:47
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

'''
## Specify which model to be used, all models are stored in 'models' 
'''
model=dict(backbone='LinearRegression3D')
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset=dict(backbone='Density_Cloud',datapath='')
test_dataset=dict(backbone='Density_Cloud',datapath='')
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config=dict(work_dir='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/LR3D_1',logfile='log.log')
'''
## Specify the checkpoint configuration
'''
checkpoint_config=dict(load_from='',resume_from='',save_inter=100)
'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
loss_fn=None
# loss_fn=dict(backbone='')
'''
## Specify the Hyperparameters to be used
'''
hyperpara=dict(epochs=5000,batch_size=400,inputshape=[1,3,10,10,40])
'''
## Specify the lr as well as its config, the lr will be optimized using torch.optim.lr_scheduler.ReduceLROnPlateau()
'''
lr_config=dict(init=0.0002,patience=50)
'''
## Specify the GPU config and DDP
'''
gpu_config=dict(distributed=True,gpuid=0)