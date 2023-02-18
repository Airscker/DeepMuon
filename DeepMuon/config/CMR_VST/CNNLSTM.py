'''
Author: airscker
Date: 2023-01-31 09:01:02
LastEditors: airscker
LastEditTime: 2023-02-18 13:13:01
Description: NULL

Copyright (C) 2023 sby Airscker(Yufeng), All Rights Reserved.
'''

'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='Dense4012FrameRNN', params=dict(
    n_classes=2, input_shape=(3, 128, 128), seq_dropout=0.1, pretrained=False))
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/workdir/annotations/4ch_cine_0.994_bin_train.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=False,
                modalities=['4ch'],
                model='LSTM',
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(210, 210)),
                                  dict(type='Resize', size=(128, 128)),
                                  dict(type='Random_rotate',
                                       range=30, ratio=0.3),
                                  ]))
test_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/workdir/annotations/4ch_cine_0.994_bin_test.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=False,
                modalities=['4ch'],
                model='LSTM',
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(210, 210)),
                                  dict(type='Resize', size=(128, 128)),
                                  dict(type='Random_rotate',
                                       range=30, ratio=0.3),
                                  ]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/JoyceW/VST_fusion_dataset/DM_workdir/CNNLSTM/4ch_bin_1.826', logfile='log.log')
'''
Specify the checkpoint configuration
'''
checkpoint_config = dict(
    load_from='', resume_from='', save_inter=50)
'''
Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
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
fsdp_parallel = dict(enabled=False, min_num_params=1e6)
