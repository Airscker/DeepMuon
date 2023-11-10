'''
Author: airscker
Date: 2023-10-03 14:19:48
LastEditors: airscker
LastEditTime: 2023-11-03 21:17:06
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
from .SmilesGraphUtils.crystal_featurizer import one_hot_decoding,one_hot_encoding,smiles_to_atom_bond_graph
from .SmilesGraphUtils.graph_operation import CombineGraph

class AtomMasking(Dataset):
    def __init__(self,
                 datapath,
                 size:int=10000,
                 full_dataset=False,
                 mask_ratio: float = None,
                 preprocess: bool=False,
                 randomize: bool = False,
                 repeat_masking: int=1,
                 mode: str = 'train',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.size=size
        self.mode=mode
        self.mask_ratio = mask_ratio

        self.full_dataset = full_dataset
        self.repeat_masking = repeat_masking
        self.dataset_iter=0
        self.dataset_iter_count=0

        self.dataset = []
        self.preprocess=preprocess
        self.atom_featurizer=CanonicalAtomFeaturizer(atom_data_field='atom_feat',alltable=True,encode_unknown=True)
        self.bond_featurizer=CanonicalBondFeaturizer(bond_data_field='bond_feat',self_loop=False)
        if not preprocess:
            self._load_preprocessed(datapath)
            if mode == 'train':
                self.dataset = self.dataset[:int(len(self.dataset) * 0.8)]
            else:
                self.dataset = self.dataset[int(len(self.dataset) * 0.8):]
        else:
            self._load_smiles(datapath)
            self.size=len(self.dataset)
        
        self.dataset_size = len(self.dataset)
        if self.mode=='train':
            if self.preprocess:
                self.interation_length=len(self.dataset)
            else:
                self.interation_length=int(self.size*0.8)
        else:
            # self.interation_length=len(self.dataset)
            if self.preprocess:
                self.interation_length=len(self.dataset)
            else:
                self.interation_length=int(self.size*0.2)
    def periodic_index(self,index):
        return index-self.dataset_size*(index//self.dataset_size)
    def _load_preprocessed(self,datapath:str):
        self.dataset=np.load(datapath,allow_pickle=True)
        if not self.full_dataset:
            self.dataset=self.dataset[:self.size]
        else:
            print('Full dataset loaded!')
    def _load_smiles(self,datapath:str):
        with open(datapath,'r') as f:
            data=f.readlines()
        if self.mode=='train':
            data=data[:int(len(data)*0.8)]
        else:
            data=data[int(len(data)*0.8):]
        print(f'Loaded {len(data)} SMILES from {datapath}, mode: {self.mode}')
        for i in range(len(data)):
            data[i]=data[i].split('\n')[0]
            try:
                for _ in range(self.repeat_masking):
                    graph=smiles_to_bigraph(data[i],False,self.atom_featurizer,self.bond_featurizer)
                    if graph is not None:
                        masked_graph, _, _,full_graph_atom= self.mask_atom(graph)
                        self.dataset.append([masked_graph,full_graph_atom])
            except:
                print(f'Error occured with SMILES: {data[i]}')
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
        return self.interation_length
    def __getitem__(self, index):
        '''
        ATTENTION: AFTER MASKING, THE GRAPH IS CHANGED!
        I.E.: masked_graph==graph
        '''
        # if self.full_dataset and self.mode=='train':
        if self.full_dataset:
            index=self.periodic_index(index+self.size*self.dataset_iter)
            self.dataset_iter_count+=1
            if self.dataset_iter_count==self.interation_length:
                self.dataset_iter+=1
        if not self.preprocess:
            graph=np.load(self.dataset[index],allow_pickle=True).item()
            masked_graph, _, _,full_graph_atom= self.mask_atom(graph)
            return masked_graph,full_graph_atom
        else:
            masked_graph,full_graph_atom=self.dataset[index]
            return masked_graph,full_graph_atom

class MolSpaceDataset(Dataset):
    def __init__(self,
                 smiles_path:str='',
                 dataset_path:str='',
                 combine_graph:bool=False,
                 pred_ce:bool=True,
                 mode:str='train',
                 target:str='CE (%)',
                 add_self_loop:bool=False,
                 shuffle:bool=False,
                 basical_encode:bool=False,
                 add_Hs:bool=False,
                 ) -> None:
        super().__init__()
        self.smiles_path=smiles_path
        self.dataset_path=dataset_path
        self.combine_graph=combine_graph
        self.pred_ce=pred_ce
        self.mode=mode
        self.target=target
        self.add_self_loop=add_self_loop
        self.shuffle=shuffle
        self.basical_encode=basical_encode
        self.add_Hs=add_Hs
        self.atom_featurizer=CanonicalAtomFeaturizer(atom_data_field='atom_feat',alltable=True,encode_unknown=True)
        self.bond_featurizer=CanonicalBondFeaturizer(bond_data_field='bond_feat',self_loop=False)
        if pred_ce:
            self._load_ce_smiles(smiles_path)
            self._load_ce_dataset(dataset_path)
        else:
            self._load_smiles(smiles_path)
            self._load_dataset(dataset_path)
    def _load_smiles(self,smiles_info,smiles_info_col=['Abbreviation','Smiles']):
        self.smiles=pd.read_csv(smiles_info,index_col=smiles_info_col[0]).to_dict()[smiles_info_col[1]]
    def _load_dataset(self,sample_info):
        sample=pd.read_csv(sample_info)
        # if start is None:
        #     start=0
        # if end is None:
        #     end=len(sample)
        # sample=sample[start:end]
        composition=sample['IL'].tolist()
        cation=sample['cation'].tolist()
        anion=sample['anion'].tolist()
        solubility=sample['x_CO2'].tolist()
        temperature=sample['T (K)'].tolist()
        pressure=sample['P (bar)'].tolist()
        all_data=list(zip(composition,cation,anion,temperature,pressure,solubility))
        comp_groups={}
        for i in range(len(all_data)):
            if all_data[i][0] not in comp_groups.keys():
                comp_groups[all_data[i][0]]=[all_data[i][1:]]
            else:
                comp_groups[all_data[i][0]].append(all_data[i][1:])
        self.dataset=[]
        for comp in comp_groups.keys():
            sub_group=comp_groups[comp]
            if self.mode=='train':
                sub_group=sub_group[:int(len(sub_group)*0.8)]
            else:
                sub_group=sub_group[int(len(sub_group)*0.8):]
            for i in range(len(sub_group)):
                cation_graph=self.generate_graph(self.smiles[sub_group[i][0]])
                anion_graph=self.generate_graph(self.smiles[sub_group[i][1]])
                if cation_graph is None or anion_graph is None:
                    continue
                else:
                    if self.basical_encode:
                        combined_graph=[cation_graph,anion_graph]
                    else:
                        combined_graph=CombineGraph([cation_graph,anion_graph],add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    self.dataset.append([combined_graph,sub_group[i][2],sub_group[i][3],sub_group[i][4]])
        if self.shuffle:
            random.shuffle(self.dataset)
    def _load_ce_smiles(self,smiles_info):
        salt=pd.read_excel(smiles_info,sheet_name='Salt',index_col='Salt').to_dict()['SMILES']
        solvent=pd.read_excel(smiles_info,sheet_name='Molecule',index_col='Solvent').to_dict()['SMILES']
        new_solv={}
        for key in solvent.keys():
            new_solv[key.split(' (')[0]]=solvent[key]
        new_salt={}
        for key in salt.keys():
            new_salt[key.split(' (')[0]]=salt[key]
        self.smiles={**new_salt,**new_solv}
    def _load_ce_dataset(self,sample_info):
        all_data=pd.read_excel(sample_info)
        if self.mode=='train':
            all_data=all_data[:int(len(all_data)*0.8)]
        else:
            all_data=all_data[int(len(all_data)*0.8):]
        comp_salt=all_data['Salt'].tolist()
        comp_solv=all_data['Solvent'].tolist()
        composition=[]
        for i in range(len(comp_salt)):
            salt=comp_salt[i].split(',')
            solv=comp_solv[i].split(',')
            composition.append(salt+solv)
        solv_mol=all_data[['Solvent 1 mol/L','Solvent 2 mol/L','Solvent 3 mol/L']].to_numpy()
        salt_mol=all_data[['Salt 1 mol/L','Salt 2 mol/L','Salt 3 mol/L']].to_numpy()
        env_info=all_data[['FC','OC','FO','InOr','F','sF','aF','O','sO','aO','C','sC','aC']].to_numpy().tolist()
        ColumbicEfficiency=all_data[self.target].tolist()
        self.dataset=[]
        for i in range(len(composition)):
            graphs=[]
            for j in range(len(composition[i])):
                graph=self.generate_graph(self.smiles[composition[i][j]])
                if graph is None:
                    graphs=[]
                    break
                else:
                    graphs.append(graph)
            if len(graphs)!=0:
                if self.basical_encode:
                    graphs=list(map(list,zip(*graphs)))
                    combined_atom_graph=CombineGraph(graphs[0],add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    combined_bond_graph=CombineGraph(graphs[1],add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    self.dataset.append([combined_atom_graph,combined_bond_graph,np.concatenate([solv_mol[i],salt_mol[i]]).tolist(),ColumbicEfficiency[i]])
                else:
                    if self.combine_graph:
                        combined_graph=CombineGraph(graphs,add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    else:
                        combined_graph=dgl.batch(graphs)
                    self.dataset.append([combined_graph,np.concatenate([solv_mol[i],salt_mol[i]]).tolist(),ColumbicEfficiency[i]])
                # self.dataset.append([combined_graph,env_info[i],ColumbicEfficiency[i]])
        if self.shuffle:
            random.shuffle(self.dataset)
    def generate_graph(self,smiles):
        try:
            if self.basical_encode:
                atom_graph,bond_graph=smiles_to_atom_bond_graph(smiles,self.add_Hs)
                if atom_graph is None or bond_graph is None:
                    return None
                else:
                    return atom_graph,bond_graph
            else:
                graph=smiles_to_bigraph(smiles,
                                        add_self_loop=self.add_self_loop,
                                        node_featurizer=self.atom_featurizer,
                                        edge_featurizer=self.bond_featurizer)
                return graph
        except:
            return None
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, index):
        if self.basical_encode:
            cation_atom_graph=self.dataset[index][0][0][0]
            cation_bond_graph=self.dataset[index][0][0][1]
            anion_atom_graph=self.dataset[index][0][1][0]
            anion_bond_graph=self.dataset[index][0][1][1]
            ratio=self.dataset[index][1:3]
            label=self.dataset[index][3]
            return cation_atom_graph,cation_bond_graph,anion_atom_graph,anion_bond_graph,ratio,label
        else:
            graph=self.dataset[index][0]
            ratio=self.dataset[index][1]
            label=self.dataset[index][2]
            return graph,ratio,label

def collate_molspacev2(samples):
    samples=list(map(list,zip(*samples)))
    cation_atom_graph,cation_bond_graph,anion_atom_graph,anion_bond_graph,ratios,labels=samples
    ratios = torch.FloatTensor(ratios)
    labels = torch.FloatTensor(labels)
    data={}
    data['cation_atom_graphs']=dgl.batch(cation_atom_graph)
    data['cation_bond_graphs']=dgl.batch(cation_bond_graph)
    data['anion_atom_graphs']=dgl.batch(anion_atom_graph)
    data['anion_bond_graphs']=dgl.batch(anion_bond_graph)
    data['add_features']=ratios
    return data, labels

def collate_molspace(samples):
    samples=list(map(list,zip(*samples)))
    graphs = samples[0]
    ratios = torch.FloatTensor(samples[1])
    labels = torch.FloatTensor(samples[2])
    data={}
    data['graphs']=graphs
    data['add_features']=ratios
    return data, labels

def collate_atom_masking(samples):
    samples=list(map(list,zip(*samples)))
    graphs = samples[0]
    # mask_idx = torch.from_numpy(np.concatenate(samples[1], axis=0))
    # masked_atom_feature = samples[1]
    atom_feature = samples[1]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.concatenate(atom_feature,dim=0)