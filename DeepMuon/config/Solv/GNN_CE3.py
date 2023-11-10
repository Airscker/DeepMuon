'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-10-31 17:20:16
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
model = dict(backbone='MolSpaceGNN',pipeline='molspacepipe',
             params=dict(depth=3,
                 num_angular=9,
                 num_radial=9,
                 cutoff=5,
                 smooth_cutoff=5,
                 learnable_rbf=True,
                 atom_embedding_dim=128,
                 bond_embedding_dim=128,
                 angle_embedding_dim=128,
                 atom_num_embedding=100,
                 atomconv_hidden_dim=[256],
                 atomconv_dropout=0,
                 bondconv_hidden_dim=[256],
                 bondconv_dropout=0,
                 angleconv_hidden_dim=[256],
                 angleconv_dropout=0,))

train_dataset = dict(backbone='MolSpaceDataset',collate_fn='collate_molspacev2',
                     params=dict(smiles_path='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                        dataset_path='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                        combine_graph=True,
                        pred_ce=True,
                        mode='train',
                        target='LCE',
                        add_self_loop=False,
                        shuffle=False,
                        basical_encode=True,
                        add_Hs=False,))
test_dataset = dict(backbone='MolSpaceDataset',collate_fn='collate_molspacev2',
                    params=dict(smiles_path='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                        dataset_path='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                        combine_graph=True,
                        pred_ce=True,
                        mode='train',
                        target='LCE',
                        add_self_loop=False,
                        shuffle=False,
                        basical_encode=True,
                        add_Hs=False,))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/ColumbicEfficiency/MolSP001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=4)

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