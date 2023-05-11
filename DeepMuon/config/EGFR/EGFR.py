'''
Author: airscker
Date: 2022-09-20 22:24:05
LastEditors: airscker
LastEditTime: 2023-03-29 00:23:18
Description: Configuration of Hailing 1TeV MLP3_3D_Direct Model

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''
## Specify which model to be used, all models are stored in 'models' 
'''
search_config=dict(search_space= {'augment_ratio1': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2]}},
                   exp_name='EXP',
                   concurrency=2,
                   trial_number=10,
                   port=14001,
                   tuner='TPE')
search_params=dict(augment_ratio1=0.3)
model = dict(backbone='ResNet50_cls_rec',pipeline='classify',params=dict(num_classes=2))
'''
## Specify the dataset to load the data, all dataset are stored in 'dataset'
'''
train_dataset = dict(backbone='EGFR_NPY', 
                     params=dict(img_dataset='/home/dachuang2022/Yufeng/data_code_egfr_0316/img_dataset/imgs_train.pkl',
                                augment=False,
                                augment_ratio=[search_params['augment_ratio1'],0.6],
                                augment_pipeline=[dict(type='flip'),
                                                dict(type='rotate',angle_range=180),
                                                dict(type='bright',light_range=(0.8,1.1))]))
test_dataset = dict(backbone='EGFR_NPY', params=dict(img_dataset='/home/dachuang2022/Yufeng/data_code_egfr_0316/img_dataset/imgs_val.pkl', augment=False))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(
    work_dir='/home/dachuang2022/Yufeng/EGFR/work_dir/NNHSEXP01')
'''
## Specify the checkpoint configuration
'''
checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
loss_fn = dict(backbone='CLS_REC_KLD')
# loss_fn=dict(backbone='MultiClassFocalLossWithAlpha',params=dict(alpha=[0.3,0.6],gamma=2))
evaluation = dict(metrics=['f1_score', 'confusion_matrix','AUC',
                           'every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='AUC'))
'''
## Specify the Hyperparameters to be used
'''
hyperpara = dict(epochs=20, batch_size=16)
'''
optimizer
'''
optimizer = dict(backbone='AdamW', params=dict(
    lr=0.0001, weight_decay=0.05, betas=(0.9, 0.999)))
'''
scheduler
'''
# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=10,eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(
    mode='min', factor=0.5, patience=50))
'''
## Specify the GPU config and DDP
'''
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=0.01, double_precision=False)
