'''
Author: airscker
Date: 2022-09-17 18:11:14
LastEditors: airscker
LastEditTime: 2022-09-21 11:09:43
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
def pattern_data_1T(event,shape=(10,10,40,3)):
    """## Convert the Original Hailing Data into Pattern Image with specified shape
    - Args:
        - event: Single Hailing original data
        - shape: The shape of the pattern data. Defaults to (10,10,40,3) for 1TeV data, or (10,10,50,3) for 10TeV data
    - Returns:
        - Converted Pattern Image with specified shape, dtype: nparray
    """
    pattern=np.zeros(shape)
    for i in range(len(event)):
        pattern[int(event[i][0])][int(event[i][1])][int(event[i][2])]=event[i][3:]
    return pattern

class HailingDataset_Fusion(Dataset):
    def __init__(self,datapath='../Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV Data
        - Args:
            - datapath: The datapath of preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [6,], dtype: nparray -> torch.tensor, info: [x,y,z,px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=torch.from_numpy(self.origin_data[index][0])
        label=torch.from_numpy(self.origin_data[index][1])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()

class HailingDataset_Pos(Dataset):
    def __init__(self,datapath='../Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args:
            - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [x,y,z]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=torch.from_numpy(self.origin_data[index][0])
        label=torch.from_numpy(self.origin_data[index][1][:3])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()

class HailingDataset_Direct(Dataset):
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
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=torch.from_numpy(self.origin_data[index][0])
        label=torch.from_numpy(self.origin_data[index][1][3:])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()

class HailingData_Init:
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/validate_norm.pkl',output='Hailing_1TeV_val_data.pkl',shape=(10,10,40,3)):
        """## Convert the Original Hailing Data into a list of Pattern Images(dtype: nparray -> torch.tensor) as well as Position-Direction(dtype: nparray -> torch.tensor)
        - Args:
            - datapath: The data path of the original Hailing data. Defaults to './Hailing-Muon/data/1TeV/validate_norm.pkl'.
            - output: The output path of the converted Hailing pattern image data: [description]. Defaults to 'Hailing_1TeV_val_data.pkl'.
            - shape: The shape of the Hailing pattern image data. Defaults to (10,10,40,3) for 1TeV data and (10,10,50,3) for 10TeV data available.
        """
        self.datapath=datapath
        self.origin_data=None
        self.output=output
        self.shape=shape
        self.__Init()
    def __Init(self):
        '''Used to save pattern imgs, shape[10,10,40/50,3] and label[6,]'''
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()
        bar=tqdm(range(len(self.origin_data)),mininterval=1)
        bar.set_description('Loading dataset')
        newdata=[]
        for i in bar:
            if len(self.origin_data[i][0].shape)==2:
                label=np.array(self.origin_data[i][1])
                label[:3]=label[:3]/1000.0
                newdata.append([np.array(pattern_data_1T(self.origin_data[i][0],shape=self.shape)),label])
        with open(self.output,'wb')as f:
            pkl.dump(newdata,f)
        f.close()
        print(f'file saved as {self.output}')
# data=HailingData_Init(datapath='../Hailing-Muon/data/1TeV/validate_norm.pkl',output='../Hailing-Muon/data/1TeV/Hailing_1TeV_val_data_1k.pkl')
# data=HailingData_Init(datapath='../Hailing-Muon/data/1TeV/train_norm.pkl',output='../Hailing-Muon/data/1TeV/Hailing_1TeV_train_data_1k.pkl')
# HailingDataset_Direct()