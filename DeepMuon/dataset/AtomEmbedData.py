'''
Author: airscker
Date: 2023-10-03 14:19:48
LastEditors: airscker
LastEditTime: 2023-10-16 21:17:41
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import time
import random
import pickle as pkl
import numpy as np
import pandas as pd
from typing import Any

import dgl
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from tqdm import tqdm
from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from .SmilesGraphUtils.molecular_graph import smiles_to_bigraph
from .SmilesGraphUtils.crystal_featurizer import one_hot_decoding

# allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list':
#     list(range(1, 119)),
#     'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list': [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list': [
#         Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
#         Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
#         Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2,
#         Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds': [
#         Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs': [  # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }


class AtomMasking(Dataset):
    def __init__(self,
                 datapath,
                 size:int=10000,
                 mask_ratio: float = None,
                 randomize: bool = False,
                 mode: str = 'train',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.size=size
        self.mask_ratio = mask_ratio
        self.dataset = []
        # self._load_smiles(datapath)
        self._load_preprocess(datapath)
        self.atom_featurizer=CanonicalAtomFeaturizer(atom_data_field='atom_feat',alltable=True,encode_unknown=True)
        self.bond_featurizer=CanonicalBondFeaturizer(bond_data_field='bond_feat',self_loop=False)
        # np.random.seed(0)
        if randomize:
            np.random.shuffle(self.dataset)
        if mode == 'train':
            self.dataset = self.dataset[:int(len(self.dataset) * 0.8)]
        else:
            self.dataset = self.dataset[int(len(self.dataset) * 0.8):]
    def _load_preprocess(self,datapath:str):
        self.dataset=np.load(datapath,allow_pickle=True)[:self.size]
    def _load_smiles(self,datapath:str):
        # mol_list=[]
        with open(datapath,'r') as f:
            data=f.readlines()
        for i in range(len(data)):
            data[i]=data[i].split('\n')[0]
            # try:
            #     mol_list.append(Chem.MolFromSmiles(data[i]))
            # except:
            #     print(f'Error occured with SMILES: {data[i]}')
            #     continue
        self.dataset=data[:min(self.size,len(data))]
    def _load_gdb(self, datapath: str) -> Any:
        for roots,dirs,files in os.walk(datapath):
            if len(dirs)==0:
                for file in files:
                    if file.endswith('.txt'):
                        with open(os.path.join(roots,file),'r') as f:
                            data=f.readlines()
                        for i in range(len(data)):
                            data[i]=data[i].split('\n')[0]
                            try:
                                self.dataset.append([data[i],Chem.MolFromSmiles(data[i])])
                            except:
                                print(f'Error occured with SMILES: {data[i]}')
                                continue
            # try:
            #     _, graph = self.smiles2graph(data[i], self.bidirectional)
            # except:
            #     print(f'Error occured with SMILES: {data[i]}')
            #     continue
            # graph, mask_index, masked_atom_feature = self.mask_atom(graph)
            # self.dataset.append((graph, mask_index, masked_atom_feature))

    def _load_qm9(self, datapath: str):
        if not self.preprocessed:
            data=pd.read_csv(datapath)['smiles'][3:].values
            for i in range(len(data)):
                try:
                    self.dataset.append([data[i],Chem.MolFromSmiles(data[i])])
                except:
                    print(f'Error occured with SMILES: {data[i]}')
                    continue
        else:
            with open(datapath, 'rb') as f:
                data = pkl.load(f)
            for key in data.keys():
                graph = data[key][-1]
                graph, _, masked_atom_feature = self.mask_atom(graph)
                self.dataset.append([graph,masked_atom_feature])

    def mask_atom(self, graph: dgl.DGLGraph):
        atom_num = graph.number_of_nodes()
        node_feat=graph.ndata['atom_feat']
        mask_num = max(1, int(atom_num * 0.15))
        mask_index = np.random.choice(atom_num, mask_num, replace=False)
        full_atom_feature = torch.argmax(node_feat[:,:118],dim=1)+1
        masked_atom_feature=full_atom_feature[mask_index]

        node_feat[mask_index] = 0
        graph.ndata['atom_feat']=node_feat
        return graph, mask_index, torch.LongTensor(masked_atom_feature), torch.LongTensor(full_atom_feature)

    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, index):
        '''
        ATTENTION: AFTER MASKING, THE GRAPH IS CHANGED!
        I.E.: masked_graph==graph
        '''
        graph=np.load(self.dataset[index],allow_pickle=True).item()
        masked_graph, _, _,full_graph_atom= self.mask_atom(graph)
        return masked_graph,full_graph_atom


def collate_atom_masking(samples):
    samples=list(map(list,zip(*samples)))
    graphs = samples[0]
    # mask_idx = torch.from_numpy(np.concatenate(samples[1], axis=0))
    # masked_atom_feature = samples[1]
    atom_feature = samples[1]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.concatenate(atom_feature,dim=0)
