'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-09-25 17:22:13
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''
Specify which model to be used, all models are stored in 'models'
'''
model = dict(backbone='screening_model', pipeline='classify',params=dict(num_classes=11))
'''
Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/demo_ann.txt',
                mask_ann='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/demo_mask_ann.pkl',
                fusion=False,
                frame_interval=2,
                augment_pipeline=[dict(type='HistEqual'),
                                dict(type='Padding', size=(120, 120)),
                                dict(type='Random_rotate',range=180, ratio=0.3),
                                dict(type='Resize', size=(224, 224)),
                                dict(type='Batch_norm',mean=151.28,std=69.32),
                                dict(type='AddRandomNumber',range=0.1)]))
test_dataset = dict(
    backbone='NIIDecodeV2',
    params=dict(ann_file='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/demo_ann.txt',
                mask_ann='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/demo_mask_ann.pkl',
                fusion=False,
                frame_interval=2,
                augment_pipeline=[dict(type='HistEqual'),
                                  dict(type='Batch_norm',mean=154.5,std=66.62),
                                  dict(type='Padding', size=(120, 120)),
                                  dict(type='Resize', size=(224, 224))]))
'''
Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(work_dir='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/')
'''
Specify the checkpoint configuration
'''
checkpoint_config = dict(
    load_from='/home/amax/Desktop/HCM/VST/VST/SAX_DIAG_11CLS_DEMO/best_f1_score_epoch_184.pth', resume_from='', save_inter=50)
'''
Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
# loss_fn=None
loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'ConfusionMatrix',
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
hyperpara = dict(epochs=200, batch_size=1, inputshape=[1, 3, 40, 10, 10])
fsdp_parallel = dict(enabled=False, min_num_params=1e6)