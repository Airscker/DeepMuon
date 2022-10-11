'''
Author: airscker
Date: 2022-10-11 19:14:16
LastEditors: airscker
LastEditTime: 2022-10-12 00:05:27
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import numpy as np
from tqdm import tqdm
import pickle as pkl
import numba

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

@numba.jit
def gen_densityv2(vec=np.array([1,1,1]),shape=(1,10,10,40)):
    imgs=np.zeros(shape)
    for i in range(shape[3]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                imgs[:,j,k,i]=np.sum(np.cross(np.array([j,k,i]),vec)**2)
    return imgs

class Density_Cloud(Dataset):
    def __init__(self,datapath='') -> None:
        super().__init__()
    def __getitem__(self, index):
        vec=self.norm(np.random.randn(3))
        data=gen_densityv2(vec,shape=(1,10,10,40))
        return torch.from_numpy(data),torch.from_numpy(vec)
    def __len__(self):
        return 10000
    def norm(self,vec):
        return vec/np.sum(vec**2)
    