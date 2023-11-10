'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-11-05 15:47:38
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
model = dict(backbone='SolvGNNV7',pipeline='solvgnn',params=dict(in_dim=74, hidden_dim=1024, add_dim=2, mlp_dims=[1024,512], gcr_layers=2 ,n_classes=1,allow_zero_in_degree=True))

train_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_ce',
                     params=dict(pretrained_path='/data/yufeng/pretrained/model_gin/supervised_contextpred.pth',
                                 pretrain_embedding=False,
                                 pred_ce=False,
                                 smiles_info='/data/yufeng/MINES/MultiGraph/smiles.csv',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/MultiGraph/whole.csv',
                                 start=0,
                                 end=8000,
                                 mode='train',
                                 binary=True,
                                 combine_graph=False,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))
test_dataset = dict(backbone='MultiSmilesGraphData',collate_fn='collate_ce',
                    params=dict(pretrained_path='/data/yufeng/pretrained/model_gin/supervised_contextpred.pth',
                                 pretrain_embedding=False,
                                 pred_ce=False,
                                 smiles_info='/data/yufeng/MINES/MultiGraph/smiles.csv',
                                 smiles_info_col=['Abbreviation','Smiles'],
                                 sample_info='/data/yufeng/MINES/MultiGraph/whole.csv',
                                 start=8000,
                                 end=None,
                                 mode='test',
                                 binary=True,
                                 combine_graph=False,
                                 add_self_loop=False,
                                 featurize_edge=False,
                                 shuffle=False))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/CO2_SOLV/GNNV6M003')

checkpoint_config = dict(load_from='', resume_from='', save_inter=200)

loss_fn = dict(backbone='MSELoss')
evaluation = dict(metrics=['R2Value'],
                  sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=100, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)