'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-09-25 17:37:28
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
model = dict(backbone='XASMLP',pipeline='classify',
            params=dict(dim_input=100,
            classes=2,
            hidden_sizes=[256,128],
            mode='NAD',
            dropout_rate=0.7))

train_dataset = dict(backbone='ValenceDataset',params=dict(annotation='/data/yufeng/MP_dataset/Fe_train_dataset.txt',available_valences=[2,3]))
test_dataset = dict(backbone='ValenceDataset',params=dict(annotation='/data/yufeng/MP_dataset/Fe_test_dataset.txt',available_valences=[2,3]))

work_config = dict(work_dir='/home/yufeng/workdir/Fe_VAL/MLP001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='CrossEntropyLoss',params=dict(weight=torch.Tensor([5,7])))
evaluation = dict(metrics=['f1_score', 'ConfusionMatrix','every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='top_k_accuracy'))

# torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = dict(backbone='AdamW', params=dict(lr=0.0003, weight_decay=0.05, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=200, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False)