'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2023-01-30 22:18:29
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (C) 2022 by Airscker(Yufeng), All Rights Reserved.
'''
'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='VST')
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file=None,
                mask_ann=None,
                fusion=False,
                modalities=[],
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(240, 240))]))
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file=None,
                mask_ann=None,
                fusion=False,
                modalities=[],
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(240, 240))]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/VST_1', logfile='log.log')
'''
Specify the checkpoint configuration
'''
# checkpoint_config=dict(load_from='',resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/CSPP_3/Best_Performance.pth',save_inter=500)
checkpoint_config = dict(
    load_from='', resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/VST_1/Best_Performance.pth', save_inter=500)
# checkpoint_config = dict(load_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/DResMax_3/Best_Performance.pth', resume_from='', save_inter=500)
'''
Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='CrossEntropyLoss')
'''
optimizer
'''
optimizer = dict(backbone='SGD', params=dict(
    lr=0.0001, momentum=0.9, nesterov=True))
'''
scheduler
'''
scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10))
'''
Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=2000, batch_size=7500, inputshape=[1, 3, 40, 10, 10])
'''
Specify the GPU config and DDP
'''
gpu_config = dict(distributed=True, gpuid=0)
