'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-07-28 11:04:19
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=512, edge_hidden_dim=512,add_dim=0))
# model = dict(backbone='SolvGNNV3',pipeline='solvgnn',params=dict(in_dim=74, hidden_dim=256, gcr_layers=5 ,n_classes=1,allow_zero_in_degree=True))

train_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                     params=dict(information_file='',
                                 solv_file=r'E:\OneDrive\OneDrive - USTC\StonyBrook\Computational Materials\SolvGNN\MINES\data\CO2_organic\solubility_co2.csv',
                                 end=2500,ID_col='ID',info_keys=['CanonicalSMILES','Solubility_CO2'],add_self_loop=True,featurize_edge=False,shuffle=True))
test_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                    params=dict(information_file='',
                                 solv_file=r'E:\OneDrive\OneDrive - USTC\StonyBrook\Computational Materials\SolvGNN\MINES\data\CO2_organic\solubility_co2.csv',
                                 start=2500,ID_col='ID',info_keys=['CanonicalSMILES','Solubility_CO2'],add_self_loop=True,featurize_edge=False,shuffle=True))

work_config = dict(work_dir='../Hailing-Muon/MINES/GNNV2002_2_1')

checkpoint_config = dict(load_from=r'E:\OneDrive\OneDrive - USTC\Muon\Hailing-Muon\MINES\GNNV2002_2\Best_loss_epoch_378.pth', resume_from='', save_inter=100)

loss_fn = dict(backbone='MSELoss')
# evaluation = dict(metrics=['r2_score'],
#                   sota_target=dict(mode='max', target='r2_score'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-5, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=1000, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)