'''
Author: airscker
Date: 2023-10-04 13:23:27
LastEditors: airscker
LastEditTime: 2023-11-13 14:17:04
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(
    backbone='MolProperty',
    #  backbone='MolProperty',
    pipeline='molspacepipe',
    params=dict(depth=8,
                atom_dim=74,
                num_angular=9,
                num_radial=9,
                cutoff=5,
                smooth_cutoff=5,
                learnable_rbf=True,
                densenet=False,
                residual=False,
                use_message = True,
                atom_embedding_dim=1024,
                bond_embedding_dim=1024,
                angle_embedding_dim=1024,
                atom_num_embedding=100,
                atomconv_hidden_dim=[1024],
                atomconv_dropout=0,
                bondconv_hidden_dim=[1024],
                bondconv_dropout=0,
                angleconv_hidden_dim=[1024],
                angleconv_dropout=0,
                mlp_dims=[1024,512],
                mlp_out_dim=1))

train_dataset = dict(
    backbone='MolFoundationDataset',
    collate_fn='collate_molfoundation',
    num_workers=0,
    params=dict(dataset_type='qm9',
                label_col='mu',
                smiles_col='smiles',
                filepath='/data/yufeng/CollectedDataset/QM9/qm9.csv',
                preprocessed_filepath='/data/yufeng/CollectedDataset/QM9',
                add_Hs=False,
                full_atomfeat=True,
                return_bond_graph=True,
                mode='train',
                show_bar=False))
test_dataset = dict(
    backbone='MolFoundationDataset',
    collate_fn='collate_molfoundation',
    num_workers=0,
    params=dict(dataset_type='qm9',
                label_col='mu',
                smiles_col='smiles',
                filepath='/data/yufeng/CollectedDataset/QM9/qm9.csv',
                preprocessed_filepath='/data/yufeng/CollectedDataset/QM9',
                add_Hs=False,
                full_atomfeat=True,
                return_bond_graph=True,
                mode='test',
                show_bar=False))

work_config = dict(work_dir='/home/yufeng/workdir/MolDS/MolP004')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='L1Loss')
evaluation = dict(metrics=['R2Value'],
                  sota_target=dict(mode='max', target='R2Value'))

optimizer = dict(backbone='AdamW',
                 params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau',
                 params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=100, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=1,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=True)
