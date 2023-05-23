'''
Author: airscker
Date: 2023-05-23 13:46:07
LastEditors: airscker
LastEditTime: 2023-05-23 18:01:07
Description: NULLs

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

model = dict(backbone='SolvGNN',pipeline='solvgnn',params=dict())

train_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                     params=dict(information_file='/home/dachuang2022/Yufeng/SolvGNN/MINES/data/comp_data.csv',
                                 solubility_file='/home/dachuang2022/Yufeng/SolvGNN/MINES/data/cnn_data.csv',end=700))
test_dataset = dict(backbone='SmilesGraphData',collate_fn='collate_solubility',
                    params=dict(information_file='/home/dachuang2022/Yufeng/SolvGNN/MINES/data/comp_data.csv',
                                solubility_file='/home/dachuang2022/Yufeng/SolvGNN/MINES/data/cnn_data.csv',start=700))

work_config = dict(work_dir='/home/dachuang2022/Yufeng/MINES/workdir/debug')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='MSELoss')
# evaluation = dict(metrics=['f1_score', 'confusion_matrix','every_class_accuracy', 'top_k_accuracy'],
#                   sota_target=dict(mode='max', target='top_k_accuracy'))

optimizer = dict(backbone='AdamW', params=dict(lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-3, momentum=0, nesterov=False))

# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=200, batch_size=64)
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False,find_unused_parameters=False)