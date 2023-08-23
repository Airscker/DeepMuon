'''
Author: airscker
Date: 2023-05-18 13:58:39
LastEditors: airscker
LastEditTime: 2023-08-23 15:42:50
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

def CombineGraph(graphs:list[dgl.DGLGraph],add_global:bool=True,bi_direction:bool=True):
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
    
def collate_solubility(batch):
    keys = list(batch[0][0].keys())[1:]
    samples=[]
    solubilities=[]
    for info in batch:
        samples.append(info[0].values())
        solubilities.append(info[1])
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['graph'] = dgl.batch(samples[0])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+1])
        batched_sample[key] = torch.tensor(samples[i+1])
    return batched_sample,torch.Tensor(solubilities)