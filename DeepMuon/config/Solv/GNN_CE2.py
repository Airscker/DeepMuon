'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-11-04 19:17:11
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

# model = dict(backbone='SolvGNNV2',pipeline='solvgnn',params=dict(hidden_dim=256, edge_hidden_dim=512,add_dim=0))
# model = dict(backbone='MolSpace',pipeline='molpretrain',
#              params=dict(classes=1,
#                          add_dim=2,
#                          mlp_dims=[],
#                          dropout=0.0,
#                          embedding_dim=1024,
#                          num_heads=8,
#                          attn_depth=10,
#                          aggragate='sum',
#                          pretrained_path='/home/yufeng/workdir/MolPT/GINV1CE003/Best_top_k_accuracy_epoch_3.pth',
#                          freeze_gnn=False,
#                          atom_feat_dim=150,
#                          bond_feat_dim=12,
#                          gnn_layers=20,
#                          gnn_res_connection=2,
#                          ))
model = dict(backbone='MulMolSpace',pipeline='molpretrain',
             params=dict(classes=1,
                        add_dim=2,
                        atom_dim=74,
                        mlp_dims=[1024,512],
                        depth=5,
                        num_angular=9,
                        num_radial=9,
                        cutoff=5,
                        smooth_cutoff=5,
                        learnable_rbf=True,
                        atom_embedding_dim=1024,
                        bond_embedding_dim=1024,
                        angle_embedding_dim=1024,
                        atom_num_embedding=100,
                        atomconv_hidden_dim=[1024],
                        atomconv_dropout=0,
                        bondconv_hidden_dim=[1024],
                        bondconv_dropout=0,
                        angleconv_hidden_dim=[1024],
                        angleconv_dropout=0,))
train_dataset = dict(backbone='MolSpaceDataset',
                    collate_fn='collate_molspacev2',
                    params=dict(
                        # smiles_path='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                        # dataset_path='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                        smiles_path='/data/yufeng/MINES/MultiGraph/smiles.csv',
                        dataset_path='/data/yufeng/MINES/MultiGraph/whole.csv',
                        combine_graph=False,
                        pred_ce=False,
                        mode='train',
                        #  target='CE (%)',
                        target='LCE',
                        add_self_loop=False,
                        shuffle=False,
                        basical_encode=True,
                    ))
test_dataset = dict(backbone='MolSpaceDataset',collate_fn='collate_molspacev2',
                     params=dict(
                        # smiles_path='/data/yufeng/MINES/ColumbicEfficiency/mol_data.xlsx',
                        # dataset_path='/data/yufeng/MINES/ColumbicEfficiency/dataset.xlsx',
                        smiles_path='/data/yufeng/MINES/MultiGraph/smiles.csv',
                        dataset_path='/data/yufeng/MINES/MultiGraph/whole.csv',
                        combine_graph=False,
                        pred_ce=False,
                        mode='test',
                        #  target='CE (%)',
                        target='LCE',
                        add_self_loop=False,
                        shuffle=False,
                        basical_encode=True,))

work_config = dict(work_dir='/home/yufeng/workdir/MINES/CE_PT/MSPV1001')

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
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=True)
