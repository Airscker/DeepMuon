'''
Author: airscker
Date: 2023-11-07 23:58:28
LastEditors: airscker
LastEditTime: 2023-11-10 12:29:05
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import dgl
import pandas as pd
import numpy as np

from tqdm import tqdm
from .SmilesGraphUtils.crystal_featurizer import smiles_to_atom_bond_graph
from .SmilesGraphUtils.molecular_graph import smiles_to_bigraph
from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer,CanonicalBondFeaturizer

import torch
from torch.utils.data import Dataset

class MolFoundationDataset(Dataset):
    def __init__(self,
                 dataset_type:str='qm9',
                 label_col:str='mu',
                 smiles_col:str='smiles',
                 filepath:str='',
                 preprocessed_filepath:str='',
                 add_Hs:bool=False,
                 full_atomfeat:bool=True,
                 return_bond_graph:bool=True,
                 mode:str='train',
                 show_bar=False) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.label_col = label_col
        self.smiles_col = smiles_col
        self.filepath = filepath
        self.preprocessed_filepath = preprocessed_filepath
        self.add_Hs = add_Hs
        self.only_atomic_num = not full_atomfeat
        self.return_bond_graph = return_bond_graph
        self.mode=mode
        self.show_bar=show_bar

        self.atom_graphs = []
        self.bond_graphs = []
        self.smiles=[]
        self.labels={}
        if self.dataset_type == 'qm9':
            self._load_qm9()
    def _split_dataset(self,dataset):
        if self.mode=='full':
            pass
        elif self.mode=='train':
            dataset=dataset[:int(len(dataset)*0.8)]
        # elif self.mode=='val':
        #     dataset=dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        elif self.mode=='test':
            dataset=dataset[int(len(dataset)*0.8):]
        return dataset
    def _load_qm9(self):
        _chart = pd.read_csv(self.filepath,index_col=self.smiles_col)
        _chart = _chart.dropna()
        self.labels = _chart[self.label_col].to_dict()
        if os.path.exists(self.preprocessed_filepath):
            print(f'Loading preprocessed data from {self.preprocessed_filepath}')
            _atom_graphs=torch.load(os.path.join(self.preprocessed_filepath,'atom_graphs.pth'))
            _bond_graphs=torch.load(os.path.join(self.preprocessed_filepath,'bond_graphs.pth'))
            _atom_graphs=dgl.unbatch(_atom_graphs)
            _bond_graphs=dgl.unbatch(_bond_graphs)
            _pre_smiles=np.load(os.path.join(self.preprocessed_filepath,'smiles.npy'),allow_pickle=True)
            self.atom_graphs=self._split_dataset(_atom_graphs)
            self.bond_graphs=self._split_dataset(_bond_graphs)
            self.smiles=self._split_dataset(_pre_smiles)
        else:
            _chart=self._split_dataset(_chart)
            _smiles=_chart.index.tolist()
            bar = range(len(_smiles))
            if self.show_bar:
                bar=tqdm(bar,mininterval=1)
            for i in bar:
                atom_graph,bond_graph=self._generate_graph(smiles=_smiles[i])
                if atom_graph is not None:
                    self.smiles.append(_smiles[i])
                    self.atom_graphs.append(atom_graph)
                    self.bond_graphs.append(bond_graph)

    def _generate_graph(self,smiles:str):
        atom_graph,bond_graph = smiles_to_atom_bond_graph(smiles=smiles,
                                                          add_Hs=self.add_Hs,
                                                          only_atomic_num=self.only_atomic_num,
                                                          return_bond_graph=self.return_bond_graph)
        return atom_graph,bond_graph

    def __len__(self) -> int:
        return len(self.smiles)
    def __getitem__(self, index: int):
        smiles=self.smiles[index]
        label=self.labels[smiles]
        atom_graph=self.atom_graphs[index]
        bond_graph=self.bond_graphs[index]
        return atom_graph,bond_graph,label
class FoundationBasicDataset(Dataset):
    def __init__(self,
                 dataset_type: str = 'qm9',
                 label_col: str = 'mu',
                 smiles_col: str = 'smiles',
                 filepath: str = '',
                 preprocessed_filepath: str = '',
                 add_Hs: bool = False,
                 full_atomfeat: bool = True,
                 return_bond_graph: bool = True,
                 mode: str = 'train',
                 show_bar=False) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.label_col = label_col
        self.smiles_col = smiles_col
        self.filepath = filepath
        self.preprocessed_filepath = preprocessed_filepath
        self.add_Hs = add_Hs
        self.only_atomic_num = not full_atomfeat
        self.return_bond_graph = return_bond_graph
        self.mode = mode
        self.show_bar = show_bar
        self.atom_featurizer=CanonicalAtomFeaturizer(atom_data_field='atom_feat')
        self.bond_featurizer=CanonicalBondFeaturizer(bond_data_field='bond_length')
        self.atom_graphs = []
        self.smiles = []
        self.labels = {}
        if self.dataset_type == 'qm9':
            self._load_qm9()

    def _split_dataset(self, dataset):
        if self.mode == 'full':
            pass
        elif self.mode == 'train':
            dataset = dataset[:int(len(dataset) * 0.8)]
        # elif self.mode=='val':
        #     dataset=dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        elif self.mode == 'test':
            dataset = dataset[int(len(dataset) * 0.8):]
        return dataset

    def _load_qm9(self):
        _chart = pd.read_csv(self.filepath, index_col=self.smiles_col)
        _chart = _chart.dropna()
        self.labels = _chart[self.label_col].to_dict()
        if os.path.exists(self.preprocessed_filepath):
            print(
                f'Loading preprocessed data from {self.preprocessed_filepath}')
            _atom_graphs = torch.load(os.path.join(self.preprocessed_filepath,'atom_graphs_base.pth'))
            _pre_smiles = np.load(os.path.join(self.preprocessed_filepath,'smiles_base.npy'),allow_pickle=True)
            self.atom_graphs = self._split_dataset(_atom_graphs)
            self.smiles = self._split_dataset(_pre_smiles)
        else:
            _chart = self._split_dataset(_chart)
            _smiles = _chart.index.tolist()
            bar = range(len(_smiles))
            if self.show_bar:
                bar = tqdm(bar, mininterval=1)
            for i in bar:
                atom_graph = self._generate_graph(smiles=_smiles[i])
                if atom_graph is not None:
                    self.smiles.append(_smiles[i])
                    self.atom_graphs.append(atom_graph)

    def _generate_graph(self, smiles: str):
        try:
            atom_graph=smiles_to_bigraph(smiles=smiles,
                                     add_self_loop=False,
                                     node_featurizer=self.atom_featurizer,
                                     edge_featurizer=self.bond_featurizer)
        except:
            atom_graph=None
        return atom_graph
    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int):
        smiles = self.smiles[index]
        label = self.labels[smiles]
        atom_graph = self.atom_graphs[index]
        return atom_graph, None, label


def collate_molfoundation(batch):
    samples = list(map(list, zip(*batch)))
    data = {}
    atom_graphs = samples[0]
    bond_graphs = samples[1]
    labels = torch.tensor(samples[-1])
    data['atom_graphs'] = atom_graphs
    data['bond_graphs'] = bond_graphs
    return data, labels
