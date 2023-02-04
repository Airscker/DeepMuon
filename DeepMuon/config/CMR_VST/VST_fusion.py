'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-02-04 19:15:46
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='fusion_model', params=dict(
    num_classes=11, sax_weight='', lax_weight=''))
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/CNNLSTM/test_fusion.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=True,
                modalities=['sax', '4ch'],
                frame_interval=5,
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(140, 140))]))
test_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/CNNLSTM/test_fusion.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=True,
                modalities=['sax', '4ch'],
                frame_interval=5,
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(140, 140))]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/JoyceW/VST_fusion_dataset/CNNLSTM/test_fusion', logfile='log.log')
'''
Specify the checkpoint configuration
'''
# checkpoint_config=dict(load_from='',resume_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/CSPP_3/Best_Performance.pth',save_inter=500)
checkpoint_config = dict(
    load_from='', resume_from='', save_inter=50)
# checkpoint_config = dict(load_from='/data/Airscker/VST3/Hailing-Muon/work_dir/1TeV/DResMax_3/Best_Performance.pth', resume_from='', save_inter=500)
'''
Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'confusion_matrix',
                           'every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='f1_score'))
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
hyperpara = dict(epochs=100, batch_size=1, inputshape=[1, 3, 40, 10, 10])
'''
Specify the GPU config and DDP
'''
gpu_config = dict(distributed=True, gpuid=0)
