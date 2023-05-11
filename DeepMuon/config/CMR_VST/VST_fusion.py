'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-04-03 23:56:16
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='fusion_model', pipeline='classify',params=dict(
    num_classes=11,
    sax_weight='/data/JoyceW/VST_fusion_dataset/DeepMuon/checkpoints/sax_cine_11cls_0.994_best_f1_fusion_base.pth',
    lax_weight='/data/JoyceW/VST_fusion_dataset/DeepMuon/checkpoints/4ch_cine_11cls_0.994_best_f1_fusion_base.pth'))
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/DeepMuon/debug/test_fusion.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=True,
                modalities=['sax','4ch'],
                frame_interval=2,
                augment_pipeline=[dict(type='HistEqual'),
                                dict(type='Padding', size=(120, 120)),
                                dict(type='Random_rotate',range=180, ratio=0.3),
                                dict(type='Resize', size=(224, 224)),
                                dict(type='Random_Gamma_Bright',ratio=0.5, low=0.4, high=1),
                                dict(type='SingleNorm'),
                                dict(type='AddRandomNumber',range=0.1)]))
test_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/data/JoyceW/VST_fusion_dataset/CNNLSTM/test_fusion.txt',
                mask_ann='/data/JoyceW/VST_fusion_dataset/workdir/mask_ann_map.pkl',
                fusion=True,
                modalities=['4ch','4ch'],
                frame_interval=2,
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(224, 224)),
                                  dict(type='Batch_norm',mean=[153.52,155.78,0],std=[68.84,65.9,1]),]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(work_dir='/data/JoyceW/VST_fusion_dataset/DM_workdir/test_fusion')
'''
Specify the checkpoint configuration
'''
checkpoint_config = dict(
    load_from='', resume_from='', save_inter=50)
'''
Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'confusion_matrix',
                           'every_class_accuracy', 'top_k_accuracy','aucroc'],
                  sota_target=dict(mode='max', target='f1_score'))
'''
optimizer
'''
optimizer = dict(backbone='AdamW', params=dict(
    lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999)))
'''
scheduler
'''
scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10, eta_min=1e-5))
'''
Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=100, batch_size=1, inputshape=[1, 3, 40, 10, 10])
fsdp_parallel = dict(enabled=False, min_num_params=1e6)