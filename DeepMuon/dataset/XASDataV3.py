'''
Author: airscker
Date: 2024-02-01 13:59:34
LastEditors: airscker
LastEditTime: 2024-02-01 15:36:20
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import random
import pickle as pkl
import numpy as np

import dgl
import torch

from torch import nn
from torch.utils.data import ConcatDataset, Dataset
from .SmilesGraphUtils.crystal_featurizer import MPJCrystalGraphData

class XASSUMDatasetV3(Dataset):
    def __init__(self,
                 data_ann:str=None,
                 xas_type:str='XANES',
                 xas_edge:str='K',
                 shuffle:bool=True) -> None:
        super().__init__()
        self.xas_type=xas_type
        self.xas_edge=xas_edge
        self.shuffle=shuffle

    def _load_data(self,data_ann):
        with open(data_ann, 'r') as f:
            self.data_ann = f.readlines()
        if self.shuffle:
            random.shuffle(self.data_ann)
        self.xas_set = []
        self.xas_struc_map={}
        self.struc_set=[]
        for i in range(len(self.data_ann)):
            self.data_ann[i] = self.data_ann[i].split('\n')[0]
            with open(self.data_ann[i], 'rb') as f:
                data=pkl.load(f)
            self.struc_set.append([data[0]['structure'],data[2]])
            spectrums=data[1]
            for j in range(len(spectrums)):
                spec_type = str(spectrums[j]['spectrum_type'])
                spec_edge = str(spectrums[j]['edge'])
                spec_atom = str(spectrums[j]['absorbing_element'])
                if spec_type == self.xas_type and spec_edge == self.xas_edge:
                    spec_data=self._spec_len_check(spectrums[j]['spectrum'])
                    if spec_data is not None:
                        self.xas_set.append([spec_data,spec_atom])
                        self.xas_struc_map[len(self.xas_set)-1]=i

    def _spec_len_check(self,spectrum):
        spec=[spectrum.x,spectrum.y]
        if self.xas_type=='XANES':
            if len(spec[0])!=100:
                return None
        elif self.xas_type=='EXAFS':
            if len(spec[0])!=500:
                return None
        elif self.xas_type=='XAFS':
            if len(spec[0])!=600:
                return None
        else:
            return spec

    def __len__(self) -> int:
        return len(self.xas_set)

    def __getitem__(self, index):
        return self.xas_set[index],self.struc_set[self.xas_struc_map[index]]
