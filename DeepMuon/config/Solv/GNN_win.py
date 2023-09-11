'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-09-11 18:42:02
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
model = dict(backbone='SolvGNNV3',pipeline='solvgnn',params=dict(in_dim=74, hidden_dim=1024, gcr_layers=2,n_classes=1,allow_zero_in_degree=True))

train_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                     params=dict(information_file='',
                                 solv_file='/data/yufeng/MINES/data/CO2_organic/solubility_co2.csv',
                                 end=2500,ID_col='ID',info_keys=['CanonicalSMILES','Solubility_CO2'],add_self_loop=False,featurize_edge=True,shuffle=False))
test_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                    params=dict(information_file='',
                                 solv_file='/data/yufeng/MINES/data/CO2_organic/solubility_co2.csv',
                                 start=2500,ID_col='ID',info_keys=['CanonicalSMILES','Solubility_CO2'],add_self_loop=False,featurize_edge=True,shuffle=False))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/CO2_SOLV/GNNV3009')

checkpoint_config = dict(load_from='', resume_from='', save_inter=200)

loss_fn = dict(backbone='MSELoss')
evaluation = dict(metrics=['R2Value'],
                  sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=1000, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)