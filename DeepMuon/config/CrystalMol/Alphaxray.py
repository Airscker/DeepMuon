'''
Author: airscker
Date: 2023-09-15 12:17:09
LastEditors: airscker
LastEditTime: 2024-07-04 14:23:56
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='AlphaXAS',
             pipeline='xas_structure',
             params=dict(energy_dim=100,
                         hidden_dim=128,
                         num_heads=4,
                         num_channels=4,
                         hidden_channels=32,
                         expand_factor=4,
                         atom_former_blocks=8))

train_dataset = dict(backbone='alphaxasdataset',
                     collate_fn='collate_alphaxas',
                     params=dict(data_ann='/data/yufeng/data_path/trs_dataset/XAS_0-10_tr.txt',
                                 xas_type='XANES',
                                 xas_edge='K',
                                 cutoff=6.0,
                                 shuffle=True,
                                 verbose=True))
test_dataset = dict(backbone='alphaxasdataset',
                    collate_fn='collate_alphaxas',
                    params=dict(data_ann='/data/yufeng/data_path/trs_dataset/XAS_0-10_ts.txt',
                                xas_type='XANES',
                                xas_edge='K',
                                cutoff=6.0,
                                shuffle=True,
                                verbose=True))

work_config = dict(work_dir='/home/yufeng/workdir/AlphaRay/X001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=10)

loss_fn = dict(backbone='MSELoss')
# loss_fn = dict(backbone='RelativeLoss',params=dict(pos_ratio=1,sharp_ratio=0,smooth_ratio=0))
# evaluation = dict(metrics=['R2Value'],
#                   sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW',
                 params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau',
                 params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=100, batch_size=1)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=10,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=False)
