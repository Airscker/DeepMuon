'''
Author: airscker
Date: 2023-10-04 13:23:27
LastEditors: airscker
LastEditTime: 2023-10-22 16:08:16
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''


model = dict(backbone='AtomEmbedding',
             pipeline='molpretrain',
             params=dict(atom_feat_dim=150,bond_feat_dim=12,emb_dim=1024,gnn_layers=20,mlp_dims=[],res_connection=2))

train_dataset = dict(backbone='AtomMasking',
                     collate_fn='collate_atom_masking',
                     num_workers=0,
                     params=dict(datapath='/data/yufeng/CollectedDataset/SMILES/Graph_path.npy',
                                 size=200000,
                                 mask_ratio=0.15,
                                 randomize=False,
                                 mode='train',
                                 full_dataset=True,))
test_dataset = dict(backbone='AtomMasking',
                     collate_fn='collate_atom_masking',
                     num_workers=0,
                     params=dict(datapath='/data/yufeng/CollectedDataset/SMILES/Graph_path.npy',
                                 size=200000,
                                 mask_ratio=0.15,
                                 randomize=False,
                                 mode='test',
                                 full_dataset=True,))

work_config = dict(work_dir='/home/yufeng/workdir/MolPT/GINV1008')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='CrossEntropyLoss')
evaluation = dict(metrics=['f1_score', 'ConfusionMatrix', 'top_k_accuracy','AUC'],
                  sota_target=dict(mode='max', target='top_k_accuracy'))

optimizer = dict(backbone='AdamW',
                 params=dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau',
                 params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=400, batch_size=128)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False,
                       grad_acc=1,
                       grad_clip=None,
                       double_precision=False,
                       find_unused_parameters=False)
