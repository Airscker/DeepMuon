'''
Author: airscker
Date: 2023-01-28 11:34:38
LastEditors: airscker
LastEditTime: 2023-09-25 18:59:24
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
search_config=dict(search_space= {'weight1': {'_type': 'choice', '_value': [1,3,5,7,9]},
                                  'weight2': {'_type': 'choice', '_value': [1,3,5,7,9]},
                                  'dropout': {'_type': 'uniform', '_value': [0.1,0.8]},
                                  'batch_size':{'_type':'choice','_value':[32,64,128,256]},
                                  'weight_decay': {'_type': 'uniform', '_value': [0.01,0.2]},
                                  'lr': {'_type': 'uniform', '_value': [1e-5,1e-4]},
                                  },
                   exp_name='XASMLP',
                   concurrency=5,
                   trail_number=500,
                   port=14001,
                   tuner='TPE')
search_params=dict(weight1=5,weight2=3,dropout=0.1,weight_decay=0.1,lr=1e-3,batch_size=128)
model = dict(backbone='XASMLP',pipeline='classify',
             params=dict(dim_input=100,
                         classes=2,
                         hidden_sizes=[256,1024,128],
                         mode='NAD',
                         dropout_rate=search_params['dropout']))

train_dataset = dict(backbone='ValenceDataset',params=dict(annotation='/data/yufeng/MP_dataset/Fe_train_dataset.txt',available_valences=[2,3]))
test_dataset = dict(backbone='ValenceDataset',params=dict(annotation='/data/yufeng/MP_dataset/Fe_test_dataset.txt',available_valences=[2,3]))

work_config = dict(work_dir='/home/yufeng/workdir/Fe_VAL/MLP_NNHS001')

checkpoint_config = dict(load_from='', resume_from='', save_inter=50)

loss_fn = dict(backbone='CrossEntropyLoss',params=dict(weight=torch.Tensor([search_params['weight1'],search_params['weight2']])))
evaluation = dict(metrics=['f1_score', 'ConfusionMatrix','every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='top_k_accuracy'))

# torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = dict(backbone='AdamW', params=dict(lr=search_params['lr'], weight_decay=search_params['weight_decay'], betas=(0.9, 0.999)))
# optimizer = dict(backbone='SGD', params=dict(lr=1e-4, momentum=0.9, nesterov=True))

#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
# scheduler = dict(backbone='CosineAnnealingLR', params=dict(T_max=100, eta_min=1e-5))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(factor=0.5, patience=100))

hyperpara = dict(epochs=200, batch_size=search_params['batch_size'])
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=None, double_precision=False)