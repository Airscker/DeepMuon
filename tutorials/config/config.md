# Elements of `config.py`

Let's see an example first:

```python
'''
Author: airscker
Date: 2023-03-13 11:31:12
LastEditors: airscker
LastEditTime: 2023-03-13 14:20:58
Description: Configuration example comes from EGFR project
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
'''
## Specify which model to be used
'''
model = dict(filepath='~/Customized_model/Resnet50.py',backbone='ResNet50_C1',params=dict())
'''
## Specify the dataset to load the data
'''
train_dataset = dict(filepath='',backbone='EGFR_NPY', 
                     params=dict(img_dataset='~/EGFR/img_dataset/imgs_train.pkl',
                                augment=False,
                                augment_pipeline=[dict(type='add_random_number'),
                                                dict(type='flip'),
                                                dict(type='rotate',angle_range=180),
                                                dict(type='bright',light_range=(0.8,1.1))]))
test_dataset = dict(filepath='',backbone='EGFR_NPY', params=dict(img_dataset='~/EGFR/img_dataset/imgs_test.pkl', augment=False))
'''
## Specify the work_dir to save the training log and checkpoints
'''
work_config = dict(work_dir='~/EGFR/work_dir/EXP002')
'''
## Specify the checkpoint configuration
'''
checkpoint_config = dict(load_from='', resume_from='~/EGFR/work_dir/EXP002/Epoch_50.pth', save_inter=50)
'''
## Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
'''
loss_fn = dict(filepath='',backbone='CrossEntropyLoss',params=dict())
'''
## Specify the customized evaluation metrics to be used, if no metrics specified, no evaluation will be performed and model will trained to minimized the loss value
'''
evaluation = dict(metrics=['f1_score', 'confusion_matrix','AUC','every_class_accuracy', 'top_k_accuracy'],
                  sota_target=dict(mode='max', target='AUC'))
'''
## Specify the hyperparameters to be used
'''
hyperpara = dict(epochs=200, batch_size=16)
'''
Optimizer
'''
optimizer = dict(filepath='',backbone='AdamW', params=dict(lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999)))
'''
Scheduler
'''
scheduler = dict(filepath='',backbone='ReduceLROnPlateau', params=dict(mode='min', factor=0.5, patience=50))
'''
## Specify the precision options and DPP/FSDP
'''
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1, grad_clip=0.01, double_precision=False)
```
## model
- filepath: indicates the file path of your customized neural network model, you can directly omit this parameter if you have properly put it under the installation path of DeepMuon, folder `models`. Just like this:

  ```python
  model = dict(backbone='ResNet50_C1',params=dict())
  ```

  Otherwise, the config loading system will find your model within the file you have specified. That is to say, if you saved model `MODEL01` in the file `cus_models.py`, you need to give the full path of this file.

- backbone: the name of the model you want to use.

- params: indicates the parameters you want to specify when creating the instance of the neural network.

  If your model looks like this:

  ```python
  class ResNet50_C1(nn.Module):
      def __init__(self,para1,para2=None):
          pass
      def forward(self,x):
          return x
  ```

  To specify the value of `para1` as 10, you can define it like this:

  ```python
  model = dict(backbone='ResNet50_C1',params=dict(para1=10))
  ```

  You can directly omit the parameter `params` if you don't want to define anything:

  ```python
  model = dict(backbone='ResNet50_C1')
  ```

## train/test_dataset

- filepath: indicates the path of your customized `Dataset`, nothing needs to be given if you properly put it under the installation path of DeepMuon.
- backbone: the name of the `Dataset` to be used.
- params: indicates the parameters to be used to create the instance of the `Dataset`.

## work_config

- work_dir: indicates the folder name of your working space, this folder will contain all generated Tensorboard logs, JSON logs, text logs, checkpoints, and even evaluated results.

## checkpont_config

- load_from: indicates the checkpoint to be used when initializing the neural network's parameters.
- resume_from: indicated the checkpoint to be resumed to train the model.
- save_inter: interval number of epochs to save the checkpoint of model parameters.

## loss_fn

- filepath: indicates the path of your customized loss function, you can omit it when your loss function was properly saved under the installation path of DeepMuon.
- backbone: the name of the loss function you want to use.
- params: parameters to be used to initialize the loss function.

## evaluation

- metrics: the list of the names of evaluation metrics to be used.
- sota_target: indicates the optimized target, if omitted the model will be optimized to minimize the loss value of the testing dataset.
  - mode: only 'min'/'max' allowed, indicates whether to minimize or maximize the evaluation metric value specified by `target`.
  - target: the name of the evaluation metric to be optimized, that is to say, the standard of estimating the model's best performance and checkpoint will according to the metric specified by `target`.

## hyper_para

- epochs: the number of epochs to be trained.
- batch_size: the number of samples to be used within one iteration of model training.

## optimizer

- filepath: the path of your customized optimizer, can be omitted if you have properly put it under the installation path of DeepMuon.
- backbone: the name of the optimizer to be used.
- params: parameters to be used to initialize the optimizer.

## scheduler

- filepath: the path of your customized scheduler, can be omitted if you have properly put it under the installation path of DeepMuon.
- backbone: the name of the scheduler to be used.
- params: parameters to be used to initialize the scheduler.

## fsdp_parallel

Typically omitted, unless you want to train large models but your platform can't handle it using the DDP algorithm.

- enabled: whether to use the FSDP algorithm to train really large models.
- min_num_params: minimum number of parameters distributed on every GPU.

## optimize_config

- fp16: whether to enable mixed precision training.
- grad_acc: the number of accumulation steps of gradient, 1 means no accumulation. It's useful when your GPU restricts your `batch_size`.
- grad_clip: the maximum value of the allowed gradient when backpropagating gradients. If `None` is given this operation will be deprecated.
- double_precision: whether to enable double precision (64-bit tensors will be used) model training.