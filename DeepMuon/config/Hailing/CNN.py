
model=dict(backbone='CNN1')

train_dataset=dict(backbone='HailingDataset_Direct',datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_4Sigma45_train60k.pkl')

test_dataset=dict(backbone='HailingDataset_Direct',datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_4Sigma45_test20k.pkl')

work_config=dict(work_dir='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/CNN6',logfile='log.log')
'''

checkpoint_config=dict(load_from='',resume_from='',save_inter=10)

'''
checkpoint_config=dict(load_from='',resume_from='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/CNN6/Best_Performance.pth',save_inter=10)

loss_fn=None

hyperpara=dict(epochs=500,batch_size=800,inputshape=[1,3,10,10,40])

lr_config=dict(init=0.0001,patience=100)

gpu_config=dict(distributed=True,gpuid=0)
