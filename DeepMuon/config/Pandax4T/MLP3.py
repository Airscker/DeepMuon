'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2023-02-22 19:10:34
Description: Configuration of Pandax4T-III MLP3 Model

Copyright (C) 2022 by Airscker(Yufeng), All Rights Reserved.
'''

'''
## Specify which model to be used, all models are stored in 'models'
'''
model = dict(filepath='./DeepMuon/models/Pandax4T.py',
             backbone='MLP3', params=dict())
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(filepath='', backbone='PandaxDataset', params=dict(
                     datapath='/home/dachuang2022/Yufeng/Pandax-4T-PosRec/data/IMG2D_XY.pkl'))
test_dataset = dict(filepath='', backbone='PandaxDataset', params=dict(
                    datapath='/home/dachuang2022/Yufeng/Pandax-4T-PosRec/data/IMG2D_XY_test.pkl'))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/home/dachuang2022/Yufeng/Pandax-4T-PosRec/work_dir/MLP3', logfile='log.log')
'''
## Specify the checkpoint configuration
'''
checkpoint_config = dict(
    load_from='', resume_from='/home/dachuang2022/Yufeng/Pandax-4T-PosRec/models/MLP3_3_best/Best_Performance.pth', save_inter=100)
'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
loss_fn = None
# loss_fn=dict(filepath='',backbone='',params=dict())
'''
## Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=100, batch_size=80000, inputshape=[1, 1, 17, 17])
'''
optimizer
'''
optimizer = dict(backbone='SGD', params=dict(
    lr=0.0001, momentum=0.9, nesterov=True))
'''
scheduler
'''
scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10,eta_min=1e-5))
fsdp_parallel = dict(enabled=False, min_num_params=1e6)