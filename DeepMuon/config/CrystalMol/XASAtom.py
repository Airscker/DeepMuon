'''
Author: airscker
Date: 2023-09-15 12:17:09
LastEditors: airscker
LastEditTime: 2024-02-07 00:59:50
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='XAS_Atom',
             pipeline='xas_atom',
             params=dict(xas_type='XANES', hidden_dims=[512,128], dropout=0))

train_dataset = dict(backbone='XASSUMDatasetV3',
                     collate_fn='collate_xas_atom',
                     params=dict(data_ann='/data/yufeng/XAS_train.txt',
                                 xas_type='XANES',
                                 xas_edge='K',
                                 convert_graph=False,
                                 self_loop=False,
                                 onehot_encode=True,
                                 cutoff=6.0,
                                 shuffle=True,
                                 verbose=True))
test_dataset = dict(backbone='XASSUMDatasetV3',
                    collate_fn='collate_xas_atom',
                    params=dict(data_ann='/data/yufeng/XAS_test.txt',
                                xas_type='XANES',
                                xas_edge='K',
                                convert_graph=False,
                                self_loop=False,
                                onehot_encode=True,
                                cutoff=6.0,
                                shuffle=True,
                                verbose=True))

work_config = dict(work_dir='/home/yufeng/workdir/XASAtom/Test002')

checkpoint_config = dict(load_from='', resume_from='', save_inter=100)

loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(
    metrics=['f1_score', 'ConfusionMatrix', 'top_k_accuracy', 'AUC'],
    sota_target=dict(mode='max', target='top_k_accuracy'))
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
                       find_unused_parameters=False)
