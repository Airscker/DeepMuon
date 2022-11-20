'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2022-11-20 00:15:34
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

'''
## Specify which model to be used, all models are stored in 'models' 
'''
model=dict(filepath='./DeepMuon/models/CSPP.py',backbone='CSPP',params=dict())
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset=dict(filepath='',backbone='HailingDataset_Direct2',params=dict(datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_4Sigma45_train60k.pkl'))
test_dataset=dict(filepath='',backbone='HailingDataset_Direct2',params=dict(datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_4Sigma45_test20k.pkl'))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config=dict(work_dir='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/CSPP_5',logfile='log.log')
'''
## Specify the checkpoint configuration
'''
# checkpoint_config=dict(load_from='',resume_from='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/CSPP_3/Best_Performance.pth',save_inter=500)
checkpoint_config=dict(load_from='',resume_from='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/CSPP_5/Best_Performance.pth',save_inter=500)

'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn=dict(filepath='',backbone='MSALoss',params=dict(angle_ratio=1))
'''
## Specify the Hyperparameters to be used
'''
hyperpara=dict(epochs=5000,batch_size=11000,inputshape=[1,3,10,10,40])
'''
## Specify the lr as well as its config, the lr will be optimized using torch.optim.lr_scheduler.ReduceLROnPlateau()
'''
lr_config=dict(init=0.0001,patience=500)
'''
## Specify the GPU config and DDP
'''
gpu_config=dict(distributed=True,gpuid=0)