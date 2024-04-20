'''
Author: airscker
Date: 2023-09-15 12:17:09
LastEditors: airscker
LastEditTime: 2024-04-20 16:40:41
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='XASStructureV2',
             pipeline='xas_structure',
             params=dict(energy_level=32,
                        max_n=6,
                        max_l=12,
                        cutoff = 6.0,
                        gnn_layers = 3,
                        gnn_hidden_dims = 256,
                        xas_type="XANES",
                        learnable_exp=True,
                        learnable_eps=True,))

train_dataset = dict(backbone='XASSUMDatasetV3',
                     collate_fn='collate_xas_struc',
                     params=dict(data_ann='/data/yufeng/XAS_train.txt',
                                 xas_type='XANES',
                                 xas_edge='K',
                                 self_loop=False,
                                 onehot_encode=True,
                                 cutoff=6.0,
                                 shuffle=True,
                                 verbose=True))
test_dataset = dict(backbone='XASSUMDatasetV3',
                    collate_fn='collate_xas_struc',
                    params=dict(data_ann='/data/yufeng/XAS_test.txt',
                                xas_type='XANES',
                                xas_edge='K',
                                self_loop=False,
                                onehot_encode=True,
                                cutoff=6.0,
                                shuffle=True,
                                verbose=True))

work_config = dict(work_dir='/home/yufeng/workdir/XASStrucV2/Turbo001')

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

hyperpara = dict(epochs=30, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=1,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=True)
