'''
Author: airscker
Date: 2023-09-04 22:11:50
LastEditors: airscker
LastEditTime: 2023-09-18 12:46:29
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import torch
import dgl
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset

from tqdm import tqdm

from ..tools import MultiThreadLoader
from .SmilesGraphUtils.crystal_featurizer import MPJCrystalGraphData,one_hot_encoding

class XASSUMDataset(Dataset):
    '''
    ## Load detailed XAS/Structure information of inorganic materials from preprocessed Materials Project (https://next-gen.materialsproject.org/) database.

    ### Args:
        - folder_path: The path of the folder where the preprocessed data is stored.
        - mode: The mode of the dataset, which can be `train` or `test`, the dataset size will be about 8:2.
        - num_workers: The number of threads used to load data.
        - xas_type: The type of XAS data to be loaded. The supported types include `XANES`, `EXAFS` and `XAFS`. If `None`, all types of XAS data will be loaded.
    
    ### Returns:
        - A list of data, each element of which is a list of three elements: `[graph,atom_features,spectrum]`.
            - graph: The crystal graph of the material.
            - atom_features: The features of the absorbing atom, including:
                - the one-hot encoding (optional) of the absorbing atom type,
                - the one-hot encoding (optional) of the edge type.
            - spectrum: The absorbing spectrum of the corresponding atom.
    '''
    def __init__(self,data_path:str,mode:str='train',num_workers:int=1,xas_type:str='XANES',bidirectional:bool=True,self_loop:bool=False,onehot_encode:bool=False):
        super().__init__()
        self.xas_type=xas_type
        self.spec_types={'XANES':0,'EXAFS':1,'XAFS':2}
        self.edge_types={'K':0,'L2':1,'L3':2,'L2,3':3,'M':4}
        self.bidirectional=bidirectional
        self.self_loop=self_loop
        self.onehot_encode=onehot_encode
        self.GraphFeaturizer=MPJCrystalGraphData(bidirectional=bidirectional,self_loop=self_loop,onehot_encode=onehot_encode)
        # filelist=os.listdir(folder_path)
        with open(data_path,'rb') as f:
            dataset=pkl.load(f)
        if mode=='train':
            # filelist=filelist[:int(len(filelist)*0.8)]
            dataset=dataset[:int(len(dataset)*0.8)]
        else:
            # filelist=filelist[int(len(filelist)*0.8):]
            dataset=dataset[int(len(dataset)*0.8):]
        self.dataset=dataset
        self.clean_data()
        # sum_data=[]
        # self.dataset=[]
        # for i in range(len(filelist)):
        #     filelist[i]=os.path.join(folder_path,filelist[i])
        # if num_workers==1:
        #     for i in range(len(filelist)):
        #         sum_data.append(self.load_data(filelist[i]))
        # else:
        #     sum_data=MultiThreadLoader(LoadList=filelist,ThreadNum=num_workers,LoadMethod=self.load_data)
        # for i in tqdm(range(len(sum_data)),mininterval=1):
        #     self.convert_data(sum_data[i])
        # del sum_data
    def _onehot_encode(self,data,code_length:int):
        return one_hot_encoding(data,code_length) if self.onehot_encode else [data]
    def _spectrum_type(self,spec_type:str):
        return self._onehot_encode(self.spec_types[spec_type],len(self.spec_types))
    def _edge_type(self,edge_type:str):
        return self._onehot_encode(self.edge_types[edge_type],len(self.edge_types))
    def load_data(self,path:str):
        with open(path,"rb") as f:
            data=pkl.load(f)
        print(f'{path} loaded.')
        return data
    def clean_data(self):
        new_dataset=[]
        if self.xas_type=='XANES':
            xas_len=100
        elif self.xas_type=='EXAFS':
            xas_len=500
        elif self.xas_type=='XAFS':
            xas_len=600
        for i in range(len(self.dataset)):
            if self.dataset[i][2].shape[-1]!=xas_len:
                continue
            if np.min(self.dataset[i][2][1])<-3 or np.max(self.dataset[i][2][1])>20:
                continue
            if np.max(self.dataset[i][2][1])<0.1 or np.min(self.dataset[i][2][1])>10:
                continue
            new_dataset.append(self.dataset[i])
        self.dataset=new_dataset
    def convert_data(self,data:list):
        for i in range(len(data)):
            # structure=data[i][0]["structure"]
            try:
                graph=self.GraphFeaturizer(adj_matrix=data[i][2])
            except:
                print(f'Error occured when converting {i}th data, skipped.')
                continue
            xas_set=data[i][1]
            for j in range(len(xas_set)):
                sample=xas_set[j]['spectrum'].__dict__
                if self.xas_type!=sample['spectrum_type']:
                    continue
                absorbing_atom=self._onehot_encode(self.GraphFeaturizer._get_atom_index(sample['absorbing_element']),118)
                edge=self._edge_type(sample['edge'])
                # spectrum_type=self._spectrum_type(sample['spectrum_type'])
                # spectrum_type=sample['spectrum_type']
                spectrum=np.array([sample['x'],sample['y']])
                self.dataset.append([graph,np.concatenate([absorbing_atom,edge]),spectrum])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index: int):
        graph,prompt,full_spec=self.dataset[index]
        return [graph,prompt,full_spec[1]]
    

def collate_XASSUM(batch):
    samples=list(map(list,zip(*batch)))
    graphs=dgl.batch(samples[0])
    prompt=torch.from_numpy(np.array(samples[1])).type(torch.float32)
    label=torch.from_numpy(np.array(samples[2])).type(torch.float32)
    data={'graph':graphs,'prompt':prompt}
    return data,label