'''
Author: airscker
Date: 2022-10-11 23:49:11
LastEditors: airscker
LastEditTime: 2022-10-12 00:39:04
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import numpy as np
from tqdm import tqdm
import pickle as pkl
import numba

import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)

@numba.jit
def gen_densityv2(vec=np.array([1,1,1]),shape=(1,10,10,40)):
    imgs=np.zeros(shape)
    for i in range(shape[3]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                imgs[:,j,k,i]=np.sum(np.cross(np.array([j,k,i]),vec)**2)
    return imgs

class Hailing_UNET3D(Dataset):
    def __init__(self,datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/1Tev_UNET3D_train_3S60K.pkl'):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args: 
            - datapath: The datapth of the preprocessed Hailing data
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=torch.from_numpy(np.array(self.origin_data[index][0]))
        image=torch.permute(image,(3,0,1,2))
        image[1:,:,:,:]=0.0001*image[1:,:,:,:]
        label=torch.from_numpy(self.origin_data[index][1])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()