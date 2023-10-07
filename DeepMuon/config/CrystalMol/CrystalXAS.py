'''
Author: airscker
Date: 2023-09-15 12:17:09
LastEditors: airscker
LastEditTime: 2023-09-17 12:19:19
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='CrystalXASV1',
             pipeline='crystalxas',
             params=dict(gnn_hidden_dims=1024,
                         gnn_layers=3,
                         gnn_res_connection=1,
                         feat_dim=123,
                         prompt_dim=123,
                         mlp_hidden_dims=[2048,1024,512],
                         mlp_dropout=0,
                         xas_type='XANES'))

train_dataset = dict(backbone='XASSUMDataset',
                     collate_fn='collate_XASSUM',
                     params=dict(data_path='/data/yufeng/Graph_XAS_XANES.pkl',
                                 mode='train',
                                 num_workers=10,
                                 xas_type='XANES',
                                 bidirectional=True,
                                 self_loop=False,
                                 onehot_encode=False))
test_dataset = dict(backbone='XASSUMDataset',
                    collate_fn='collate_XASSUM',
                    params=dict(data_path='/data/yufeng/Graph_XAS_XANES.pkl',
                                mode='test',
                                num_workers=10,
                                xas_type='XANES',
                                bidirectional=True,
                                self_loop=False,
                                onehot_encode=False))

work_config = dict(work_dir='/home/yufeng/workdir/CrystalXAS/GINV1005')

checkpoint_config = dict(load_from='', resume_from='', save_inter=200)

loss_fn = dict(backbone='RelativeLoss',params=dict(pos_ratio=1,sharp_ratio=1e-3,smooth_ratio=1e-5))
# evaluation = dict(metrics=['R2Value'],
#                   sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW',
                 params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau',
                 params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=200, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=1,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=True)
