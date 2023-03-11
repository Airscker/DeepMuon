# Customize configuration

In general, when we are planning our experiments, we need to design several different configurations to find out the best model architecture and hyperparameters. However, the most researchers were using basic method: One configuration, one training pipeline. That is, most of the researchers combine their parameters' settings and other static training/testing function within one file. This brings much troubles and complexness to our project management and analysis. If you want to start a new experiment with a configuration you have to write these static codes again, also when you want to change some static methods, you have to change it many times in many files.

To improve the performance and reduce the complexness of projects' management, we used configuration dependent mechanism, that is, all frequently changed hyperparameters and training/testing pipelines are given by simple command stored in a python file, and other static codes like `train()`, `test()` are separated from configurations, in this way, every time you want to start a new experiment you just need to specify the parameters you want to use rather than rewriting whole training file as before.

Here we give you an example of MLP model, trained on the dataset MINIST. After getting familiar with the customized `Dataset` and model, now we are in the last stage of experiment preparation.

```python
model = dict(backbone='MLP')
train_dataset = dict(backbone='MINIST',params=dict(root='root',download=True,train=True))
test_dataset = dict(backbone='MINIST', params=dict(root='root',download=True,train=False))
work_config = dict(work_dir='MINIST_EXP01', logfile='log.log')
checkpoint_config = dict(load_from='', resume_from='', save_inter=10)
loss_fn = dict(backbone='MSELoss')
hyperpara = dict(epochs=100, batch_size=24)
optimizer = dict(backbone='SGD', params=dict(lr=0.0001, momentum=0.9, nesterov=True))
scheduler = dict(backbone='ReduceLROnPlateau', params=dict(mode='min', factor=0.5, patience=20))
fsdp_parallel = dict(enabled=False, min_num_params=1e4)
optimize_config = dict(fp16=False, grad_acc=1,grad_clip=None, double_precision=True)
```

Here we go! Now your experiment was fully prepared, and let's start the it! See How to start it: [Start your first experiment](https://airscker.github.io/DeepMuon/tutorials/index.html#/start_exp/start_exp).