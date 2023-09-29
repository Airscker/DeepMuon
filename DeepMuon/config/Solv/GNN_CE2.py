'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-09-27 16:24:11
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
model = dict(backbone='SolvGNNV5',pipeline='solvgnn',
             params=dict(in_dim=74,
                         hidden_dim=2048,
                         add_dim=6,
                         mlp_dims=[2048,1024,512],
                         dropout_rate=0,
                         norm=True,
                         gcr_layers=22,
                         n_classes=1,
                         res_connection=True,
                         allow_zero_in_degree=True,
                         freeze_GNN=False))

train_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_solubility',
                     params=dict(pretrained_path='/data/yufeng/pretrained/model_gin/supervised_contextpred.pth',
                                 pretrain_embedding=False,
                                 pred_ce=True,
                                 smiles_info='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                                 start=0,
                                 end=80,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))
test_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_solubility',
                    params=dict(pretrained_path='/data/yufeng/pretrained/model_gin/supervised_contextpred.pth',
                                 pretrain_embedding=False,
                                 pred_ce=True,
                                 smiles_info='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                                 start=80,
                                 end=None,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/ColumbicEfficiency/GNNV5C001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=200)

loss_fn = dict(backbone='MSELoss')
evaluation = dict(metrics=['R2Value'],
                  sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=500, batch_size=32)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)