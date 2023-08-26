'''
Author: airscker
Date: 2023-05-18 13:58:39
LastEditors: airscker
LastEditTime: 2023-08-26 14:43:48
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import random
import warnings
import dgl
import os

import torch
from torch.utils.data import Dataset

from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer
from .SmilesGraphUtils.molecular_graph import mol_to_bigraph

def CombineGraph(graphs:list[dgl.DGLGraph],add_global:bool=True,bi_direction:bool=True,add_self_loop:bool=True):
    '''
    ## Combine a list of graphs into a single graph

    ### Args:
    - graphs: list of graphs, make sure every node/edge in the graph has the same feature dimension, 
        also, the node/edge feature dimension should be the same as the other graphs in the list.
    - add_global: whether to add a global node to the graph, default is `True`. 
        If enabled, every node of every subgraph in the list will be connected to the global node.
    - bi_direction: whether to add bi-directional edges between the global node (if exists) and every other node, default is `True`. 
        If enabled, the global node will be connected to every other node in the graph, and vice versa.
        If disabled, the every node in the graph will be connected to the global node but not vice versa.
    - add_self_loop: whether to add self-loop to global node in the graph, default is `True` (Self loops of subgraph nodes are not added here).
    '''
    combined_graph = dgl.batch(graphs)
    if add_global:
        combined_graph=dgl.add_nodes(combined_graph,1)
        num_node=combined_graph.num_nodes()-1
        start=[]
        end=[]
        for i in range(num_node):
            start.append(i)
            end.append(num_node)
        if bi_direction:
            for i in range(num_node):
                start.append(num_node)
                end.append(i)
        if add_self_loop:
            start.append(num_node)
            end.append(num_node)
        combined_graph=dgl.add_edges(combined_graph,torch.LongTensor(start),torch.LongTensor(end))
    return combined_graph

class SmilesGraphData(Dataset):
    def __init__(self,
                information_file=None,
                solv_file='',
                ID_col='ID',
                info_keys=['CanonicalSMILES','Solubility_CO2'],
                start=None,
                end=None,
                add_self_loop=False,
                featurize_edge=True,
                shuffle=True,
                debug=False) -> None:
        super().__init__()
        self.debug=debug
        if os.path.exists(information_file):
            self.info_list=pd.read_csv(information_file,index_col=ID_col)
        else:
            self.info_list=None
        if add_self_loop and featurize_edge:
            warnings.warn('Self looping is forbidden when edge featurizer is enabled. We will set self looping option as false.')
            add_self_loop=False
        self.shuffle=shuffle
        self.featurize_edge=featurize_edge
        self.sol_list=pd.read_csv(solv_file,index_col=ID_col)
        self.smiles=self.sol_list[info_keys[0]].to_dict()
        self.solubility=self.sol_list[info_keys[1]].to_dict()
        # self.be_salt=self.info_list['BE_Salt'].to_dict()
        # self.be_ps=self.info_list['BE_PS'].to_dict()
        # self.ip=self.info_list['IP'].to_dict()
        self.graph_data={}
        self.sol_cid_list=self.sol_list.index.to_list()
        if self.info_list is not None:
            self.info_cid_list=self.info_list.index.to_list()
        else:
            self.info_cid_list=self.sol_cid_list
        self.cid_list=list(set(self.sol_cid_list)&set(self.info_cid_list))
        self.add_features={}
        self.mol_data={}
        if self.info_list is not None:
            for cid in self.cid_list:
                self.add_features[cid]=self.info_list.loc[cid].to_numpy().tolist()
        if start is None:
            start=0
        if end is None:
            end=len(self.cid_list)
        self.cid_list=self.cid_list[start:end]
        self.generate_graph(add_self_loop)
    def featurize_bonds(self,mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())+1
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'bond_type': torch.tensor(feats).reshape(-1, 1).float()}
    def generate_graph(self,add_self_loop):
        if self.featurize_edge:
            edge_featurizer=self.featurize_bonds
        else:
            edge_featurizer=None
        for cid in self.cid_list:
            try:
                mol = Chem.MolFromSmiles(self.smiles[cid])
                self.graph_data[cid] = []
                self.graph_data[cid].append(mol_to_bigraph(mol,add_self_loop=add_self_loop,
                                                                    node_featurizer=CanonicalAtomFeaturizer(),
                                                                    edge_featurizer=edge_featurizer,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=False,
                                                                    num_virtual_nodes=0
                                                                    ))
                hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
                hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
                self.graph_data[cid].append(hba)
                self.graph_data[cid].append(hbd)
                self.graph_data[cid].append(min(hba,hbd))
                self.mol_data[cid]=mol
            except:
                self.graph_data.pop(cid)
        self.cid_list=list(self.graph_data.keys())
        if self.shuffle:
            random.shuffle(self.cid_list)

    def __len__(self):
        return len(self.graph_data)
    def __getitem__(self, index):
        sample={}
        cid=self.cid_list[index]
        if self.debug:
            sample['mol']=self.mol_data[cid]
        sample['graph']=self.graph_data[cid][0]
        sample['inter_hb']=self.graph_data[cid][3]
        # sample['be_salt']=self.be_salt[cid]
        # sample['be_ps']=self.be_ps[cid]
        # sample['ip']=self.ip[cid]
        if self.info_list is not None:
            sample['add_features']=self.add_features[cid]
        return sample,self.solubility[cid]

class MultiSmilesGraphData(Dataset):
    def __init__(self,
                smiles_info='',
                smiles_info_col=['Abbreviation','Smiles'],
                sample_info='',
                start:int=None,
                end:int=None,
                mode='train',
                add_self_loop=True,
                featurize_edge=False,
                shuffle=True) -> None:
        super().__init__()
        if add_self_loop and featurize_edge:
            warnings.warn('Self looping is forbidden when edge featurizer is enabled. We will set self looping option as false.')
            add_self_loop=False
        self.add_self_loop=add_self_loop
        self.shuffle=shuffle
        self.featurize_edge=featurize_edge
        self.mode=mode
        self.load_smiles_dict(smiles_info,smiles_info_col)
        self.load_sample_dict(sample_info,start,end)
        
    def load_smiles_dict(self,smiles_info,smiles_info_col):
        self.smiles=pd.read_csv(smiles_info,index_col=smiles_info_col[0]).to_dict()[smiles_info_col[1]]
    def load_sample_dict(self,sample_info,start,end):
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
                    combined_graph=CombineGraph([cation_graph['graph'],anion_graph['graph']],add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    self.dataset.append([combined_graph,sub_group[i][2],sub_group[i][3],sub_group[i][4]])
        if self.shuffle:
            random.shuffle(self.dataset)
    def featurize_bonds(self,mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())+1
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'bond_type': torch.tensor(feats).reshape(-1, 1).float()}
    def generate_graph(self,smiles):
        if self.featurize_edge:
            edge_featurizer=self.featurize_bonds
        else:
            edge_featurizer=None
        graph_data={}
        try:
            mol = Chem.MolFromSmiles(smiles)
            graph_data['graph']=mol_to_bigraph(mol=mol,
                                           add_self_loop=self.add_self_loop,
                                           node_featurizer=CanonicalAtomFeaturizer(),
                                           edge_featurizer=edge_featurizer,
                                           canonical_atom_order=False,
                                           explicit_hydrogens=False,
                                           num_virtual_nodes=0)
            # Hydrogne bond acceptor and donor
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            graph_data['hydrogen_bond']=[hba,hbd,min(hba,hbd)]
            graph_data['mol']=mol
        except:
            return None
        return graph_data
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        sample={}
        sample['graph']=self.dataset[index][0]
        # sample['be_salt']=self.be_salt[cid]
        # sample['be_ps']=self.be_ps[cid]
        # sample['ip']=self.ip[cid]
        sample['add_features']=self.dataset[index][1:3]
        return sample,self.dataset[index][3]

def collate_solubility(batch):
    keys = list(batch[0][0].keys())[1:]
    samples=[]
    labels=[]
    for info in batch:
        samples.append(info[0].values())
        labels.append(info[1])
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['graph'] = dgl.batch(samples[0])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+1])
    return batched_sample,torch.Tensor(labels)