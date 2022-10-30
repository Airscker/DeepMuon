'''
Author: airscker
Date: 2022-10-12 00:27:46
LastEditors: airscker
LastEditTime: 2022-10-14 20:47:25
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import pickle as pkl
import numba
import numpy as np
from tqdm import tqdm
@numba.jit
def gen_densityv2(vec=np.array([1,1,1]),shape=(1,10,10,40)):
    imgs=np.zeros(shape)
    for i in range(shape[3]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                imgs[:,j,k,i]=np.sum(np.cross(np.array([j,k,i]),vec)**2)
    return imgs

def gen_unet3d_data(path='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma87_train60k.pkl',output='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_UNET3D_3S60K.pkl'):
    with open(path,'rb')as f:
        data=pkl.load(f)
    imgs=[]
    bar=tqdm(range(len(data)),mininterval=1)
    for i in bar:
        imgs.append([data[i][0],gen_densityv2(data[i][1][3:])])
    with open(output,'wb')as f:
        pkl.dump(imgs,f)
    print(f'{output} saved')
# gen_unet3d_data(path='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma87_train60k.pkl',
#                 output='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_UNET3D_train_3S60K.pkl')
# gen_unet3d_data(path='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_Resample_3Sigma87_test20k.pkl',
#                 output='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_UNET3D_test_3S20K.pkl')
