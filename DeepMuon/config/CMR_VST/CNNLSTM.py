'''
Author: airscker
Date: 2023-01-31 09:01:02
LastEditors: airscker
LastEditTime: 2023-02-04 09:04:25
Description: NULL

Copyright (C) 2023 sby Airscker(Yufeng), All Rights Reserved.
'''

'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='Dense4012FrameRNN', params=dict(
    n_classes=11, input_shape=(3, 90, 90), seq_dropout=0.1, pretrained=False))
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/CNNLSTM/sax_cine_1.826_11cls_test.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=False,
                modalities=['sax'],
                model='LSTM',
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(90, 90))]))
test_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/CNNLSTM/sax_cine_1.826_11cls_test.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=False,
                modalities=['sax'],
                model='LSTM',
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='SingleNorm'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(90, 90))]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/data/JoyceW/VST_fusion_dataset/CNNLSTM/workdir', logfile='log.log')
'''
Specify the checkpoint configuration
'''
checkpoint_config = dict(
    load_from='', resume_from='', save_inter=500)
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
    lr=0.0001, momentum=0.9, nesterov=True, weight_decay=0.01))
'''
scheduler
'''
scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10))
'''
Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=200, batch_size=1, inputshape=[1, 3, 40, 10, 10])
'''
Specify the GPU config and DDP
'''
gpu_config = dict(distributed=True, gpuid=0)
