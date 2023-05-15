'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-05-15 13:56:04
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
model = dict(backbone='MinistModel',pipeline='classify',params=dict())

train_dataset = dict(backbone='MinistDataset',params=dict(train=True))
test_dataset = dict(backbone='MinistDataset',params=dict(train=False))

work_config = dict(work_dir='/home/dachuang2022/Yufeng/minist/workdir')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'confusion_matrix','every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='top_k_accuracy'))

# torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
# optimizer = dict(backbone='AdamW', params=dict(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999)))
optimizer = dict(backbone='SGD', params=dict(lr=1e-3, momentum=0, nesterov=False))

#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=50, batch_size=64)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False)