'''
Author: airscker
Date: 2023-05-18 13:58:39
LastEditors: airscker
LastEditTime: 2023-07-14 10:50:14
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import dgl
import os

import torch
from torch.utils.data import Dataset

from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer
from .SmilesGraphUtils.molecular_graph import mol_to_bigraph

class SmilesGraphData(Dataset):
    def __init__(self, information_file=None,solv_file='',ID_col='ID',info_keys=['CanonicalSMILES','Solubility_CO2'],start=None,end=None,add_self_loop=True) -> None:
        super().__init__()
        if os.path.exists(information_file):
            self.info_list=pd.read_csv(information_file,index_col=ID_col)
        else:
            self.info_list=None
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
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1, 1).float()}
    def generate_graph(self,add_self_loop):
        for cid in self.cid_list:
            try:
                mol = Chem.MolFromSmiles(self.smiles[cid])
                self.graph_data[cid] = []
                self.graph_data[cid].append(mol_to_bigraph(mol,add_self_loop=add_self_loop,
                                                                    node_featurizer=CanonicalAtomFeaturizer(),
                                                                    # edge_featurizer=self.featurize_bonds,
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=False,
                                                                    num_virtual_nodes=0
                                                                    ))
                hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
                hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
                self.graph_data[cid].append(hba)
                self.graph_data[cid].append(hbd)
                self.graph_data[cid].append(min(hba,hbd))
            except:
                self.graph_data.pop(cid)
        self.cid_list=list(self.graph_data.keys())

    def __len__(self):
        return len(self.graph_data)
    def __getitem__(self, index):
        sample={}
        cid=self.cid_list[index]
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