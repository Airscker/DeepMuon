'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-10-29 23:22:54
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='SolvLinear',
             pipeline='regression',
             params=dict(in_dims=13))

train_dataset = dict(
    backbone='SolvFeat',
    params=dict(datapath='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                feat=[
                    'FC', 'OC', 'FO', 'InOr', 'F', 'sF', 'aF', 'O', 'sO', 'aO',
                    'C', 'sC', 'aC', 'CE (%)'
                ],
                mode='train'))
test_dataset = dict(
    backbone='SolvFeat',
    params=dict(datapath='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                feat=[
                    'FC', 'OC', 'FO', 'InOr', 'F', 'sF', 'aF', 'O', 'sO', 'aO',
                    'C', 'sC', 'aC', 'CE (%)'
                ],
                mode='test'))

work_config = dict(
    work_dir='/home/yufeng/workdir/MINES/ColumbicEfficiency/MLP001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=4)

loss_fn = dict(backbone='MSELoss')
evaluation = dict(metrics=['R2Value'],
                  sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW',
                 params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau',
                 params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=100, batch_size=32)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=1,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=False)
