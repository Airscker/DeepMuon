'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-08-31 13:17:01
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
model = dict(backbone='SolvGNNV3',pipeline='solvgnn',
             params=dict(in_dim=74, hidden_dim=1024, add_dim=6, gcr_layers=2 ,n_classes=1,allow_zero_in_degree=True,freeze_GNN=True))

train_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_solubility',
                     params=dict(smiles_info='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                                 start=0,
                                 end=80,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))
test_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_solubility',
                    params=dict(smiles_info='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                                 start=80,
                                 end=None,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/ColumbicEfficiency/GNNV3C003')

checkpoint_config = dict(load_from='/home/yufeng/workdir/MINES/CO2_SOLV/GNNV3M001/Checkpoint/Epoch_1000.pth', resume_from='', save_inter=200)

loss_fn = dict(backbone='MSELoss')
evaluation = dict(metrics=['r2_score'],
                  sota_target=dict(mode='max', target='r2_score'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=1000, batch_size=32)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)