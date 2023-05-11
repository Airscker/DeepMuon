'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-05-11 10:41:45
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
model = dict(backbone='MLPMixer',pipeline='classify',params=dict(depth=1,dim=100,channel= 2,token_drop= 0.1,channel_drop = 0.1,classes= 10))

train_dataset = dict(backbone='ValenceDataset',params=dict(annotation=''))
test_dataset = dict(backbone='ValenceDataset',params=dict(annotation=''))

work_config = dict(work_dir='./XAS/MLPMixer_001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'confusion_matrix','every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='f1_score'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999)))

scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10, eta_min=1e-5))

fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False)