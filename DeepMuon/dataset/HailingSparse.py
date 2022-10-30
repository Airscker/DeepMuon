'''
Author: airscker
Date: 2022-10-11 19:14:16
LastEditors: airscker
LastEditTime: 2022-10-27 22:48:14
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
# import spconv.pytorch as spconv
torch.set_default_tensor_type(torch.DoubleTensor)

class Sparse_Cloud(Dataset):
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args: 
            - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=torch.from_numpy(np.array(self.origin_data[index][0]))
        image[:,:,:,1:]=0.0001*image[:,:,:,1:]
        # image=spconv.SparseConvTensor.from_dense(image)
        label=torch.from_numpy(self.origin_data[index][1][3:])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()
    