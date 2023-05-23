'''
Author: airscker
Date: 2023-05-18 13:58:39
LastEditors: airscker
LastEditTime: 2023-05-23 13:52:01
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import dgl

import torch
from torch.utils.data import Dataset

from SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer
from SmilesGraphUtils.molecular_graph import mol_to_bigraph

class SmilesGraphData(Dataset):
    def __init__(self, information_file='',solubility_file='') -> None:
        super().__init__()
        self.info_list=pd.read_csv(information_file,index_col='CID')
        self.sol_list=pd.read_csv(solubility_file,index_col='CID')
        self.smiles=self.sol_list['CanonicalSMILES'].to_dict()
        self.solubility=self.sol_list['Solubility_LiTFSI'].to_dict()
        self.be_salt=self.info_list['Be_salt'].to_dict()
        self.be_ps=self.info_list['BE_PS'].to_dict()
        self.ip=self.info_list['IP'].to_dict()
        self.graph_data={}
        self.cid_list=self.info_list.index.to_list()
        self.generate_graph()
    def generate_graph(self):
        for cid in self.smiles:
            mol = Chem.MolFromSmiles(self.smiles[cid])
            self.graph_data[cid] = []
            self.graph_data[cid].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=CanonicalAtomFeaturizer(),
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

    def __len__(self):
        return len(self.graph_data)
    def __getitem__(self, index):
        sample={}
        cid=self.cid_list[index]
        sample['graph']=self.graph_data[cid][0]
        # sample['intra_hb']=
        sample['be_salt']=self.be_salt[cid]
        sample['be_ps']=self.be_ps[cid]
        sample['ip']=self.ip[cid]
        sample['solv']=self.solubility[cid]
        return sample
    
def collate_solubility(batch):
    keys = list(batch[0].keys())[1:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['graph'] = dgl.batch(samples[0])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+1])
        batched_sample[key] = torch.tensor(samples[i+1])
    return batched_sample