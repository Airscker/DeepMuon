'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-07-24 15:32:33
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
model = dict(backbone='XASGV2',pipeline='regression',
             params=dict(mlp_drop_out=0.3))

train_dataset = dict(backbone='ValenceDatasetV2',params=dict(annotation=r'E:\MaterialsProject\XAS_SYN_TR_800k.pkl',xy_label=False))
test_dataset = dict(backbone='ValenceDatasetV2',params=dict(annotation=r'E:\MaterialsProject\XAS_SYN_TS_200k.pkl',xy_label=False))

work_config = dict(work_dir=r'E:\OneDrive\OneDrive - USTC\StonyBrook\XAS\workdir\SYN\XASGV2003')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn=dict(backbone='MSELoss')
evaluation = dict(metrics=['r2_score'],
                  sota_target=dict(mode='max', target='r2_score'))

# torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = dict(backbone='AdamW', params=dict(lr=0.0001, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=False))

#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=100, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False)