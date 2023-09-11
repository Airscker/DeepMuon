'''
Author: airscker
Date: 2023-09-04 22:11:50
LastEditors: airscker
LastEditTime: 2023-09-11 18:45:40
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from ..tools import MultiThreadLoader
from .SmilesGraphUtils.crystal_featurizer import MPJCrystalGraphData,one_hot_encoding

class XASSUMDataset(Dataset):
    '''
    ## Load detailed XAS/Structure information of inorganic materials from preprocessed Materials Project (https://next-gen.materialsproject.org/) database.

    ### Args:
        - folder_path: The path of the folder where the preprocessed data is stored.
        - num_workers: The number of threads used to load data.
        - xas_type: The type of XAS data to be loaded. The supported types include `XANES`, `EXAFS` and `XAFS`. If `None`, all types of XAS data will be loaded.
    
    ### Returns:
        - A list of data, each element of which is a list of three elements: `[graph,atom_features,spectrum]`.
            - graph: The crystal graph of the material.
            - atom_features: The features of the absorbing atom, including the one-hot encoding (optional) of the absorbing atom type,
                the one-hot encoding (optional) of the edge type and the one-hot encoding (optional) of the spectrum type.
            - spectrum: The absorbing spectrum of the corresponding atom.
    '''
    def __init__(self,folder_path:str,num_workers:int=1,xas_type:list[str]=None,bidirectional:bool=True,self_loop:bool=False,onehot_encode:bool=False):
        super().__init__()
        self.xas_type=xas_type
        self.spec_types={'XANES':0,'EXAFS':1,'XAFS':2}
        self.edge_types={'K':0,'L2':1,'L3':2,'L2,3':3,'M':4}
        self.bidirectional=bidirectional
        self.self_loop=self_loop
        self.onehot_encode=onehot_encode
        self.GraphFeaturizer=MPJCrystalGraphData(bidirectional=bidirectional,self_loop=self_loop,onehot_encode=onehot_encode)
        filelist=os.listdir(folder_path)
        self.dataset=[]
        for i in range(len(filelist)):
            filelist[i]=os.path.join(folder_path,filelist[i])
        if num_workers==1:
            for i in range(len(filelist)):
                self.dataset.append(self.load_data(filelist[i]))
        else:
            all_data=MultiThreadLoader(LoadList=filelist,ThreadNum=num_workers,LoadMethod=self.load_data)
            for i in range(len(all_data)):
                self.dataset+=all_data[i]
    def _onehot_encode(self,data,code_length:int):
        return one_hot_encoding(data,code_length) if self.onehot_encode else [data]
    def _spectrum_type(self,spec_type:str):
        return self._onehot_encode(self.spec_types[spec_type],len(self.spec_types))
    def _edge_type(self,edge_type:str):
        return self._onehot_encode(self.edge_types[edge_type],len(self.edge_types))
    def load_data(self,path:str):
        with open(path,"rb") as f:
            data=pkl.load(f)
        dataset=[]
        for i in range(len(data)):
            structure=data[i][0]["structure"]
            graph=self.GraphFeaturizer(structure)
            xas_set=data[i][1]
            for j in range(len(xas_set)):
                sample=xas_set[j]['spectrum'].__dict__
                absorbing_atom=self._onehot_encode(self.GraphFeaturizer._get_atom_index(sample['absorbing_element']),118)
                edge=self._edge_type(sample['edge'])
                spectrum_type=self._spectrum_type(sample['spectrum_type'])
                spectrum=np.array([sample['x'],sample['y']])
                dataset.append([graph,np.concatenate([absorbing_atom,edge,spectrum_type]),spectrum])
        return dataset
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, index: int):
        return self.dataset[index]
    
def collate_XASSUM(batch):
    samples=list(zip(*batch))
    graphs=samples[0]
    features=torch.from_numpy(samples[1])
    label=torch.from_numpy(samples[2])
    data={'graph':graphs,'features':features}
    return data,label