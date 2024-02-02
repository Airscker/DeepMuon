'''
Author: airscker
Date: 2023-05-18 13:58:39
LastEditors: airscker
LastEditTime: 2023-12-13 21:46:41
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import pandas as pd
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import random
import warnings
import dgl
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from .SmilesGraphUtils.molecular_graph import mol_to_bigraph
from ..models.base import GNN_feature

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
                                                                    edge_featurizer=CanonicalBondFeaturizer() if self.featurize_edge else None,
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
        # sample['be_ps']= self.be_ps[cid]
        # sample['ip']=self.ip[cid]
        if self.info_list is not None:
            sample['add_features']=self.add_features[cid]
        return sample,self.solubility[cid]

class MultiSmilesGraphData(Dataset):
    def __init__(self,
                pretrained_path='',
                pretrain_embedding=False,
                one_chart=False,
                pred_ce=True,
                smiles_info='',
                add_feat=True,
                smiles_info_col=['Abbreviation','Smiles'],
                sample_info='',
                start:int=None,
                end:int=None,
                mode='train',
                target='LCE',
                binary=False,
                combine_graph=True,
                add_self_loop=True,
                featurize_edge=False,
                shuffle=True) -> None:
        super().__init__()
        self.binary=binary
        self.add_feat=add_feat
        self.combine_graph=combine_graph
        self.target=target
        if add_self_loop and featurize_edge:
            warnings.warn('Self looping is forbidden when edge featurizer is enabled. We will set self looping option as false.')
            add_self_loop=False
        self.add_self_loop=add_self_loop
        self.shuffle=shuffle
        self.featurize_edge=featurize_edge
        self.mode=mode
        self.pretrain_embedding=pretrain_embedding
        if pretrain_embedding:
            self.pretrained_node_featurizer=GNN_feature(num_layer=5, emb_dim=300, num_tasks=1, JK='last', drop_ratio=0, graph_pooling='mean', gnn_type='gin')
            self.pretrained_node_featurizer.from_pretrained(pretrained_path)
        else:
            self.pretrained_node_featurizer=None
        if pred_ce:
            if one_chart:
                self.load_smiles_sample(smiles_info,start,end)
            else:
                self.load_smiles_dict(smiles_info,smiles_info_col)
                self.load_sample_dict(sample_info,start,end)
        else:
            self.load_smiles_dict_old(smiles_info,smiles_info_col)
            self.load_sample_dict_old(sample_info,start,end)
    def load_smiles_sample(self,chartpath,start,end):
        chart = pd.read_csv(chartpath).fillna(0)
        if self.mode=='train':
            chart=chart[:int(len(chart)*0.8)]
        else:
            chart=chart[int(len(chart)*0.8):]
        smiles=chart.to_numpy()[:,1:7].tolist()
        labels=chart['LCE'].to_numpy()
        self.dataset=[]
        for i in range(len(smiles)):
            graphs=[]
            for j in range(len(smiles[i])):
                if smiles[i][j]==0:
                    continue
                graph=self.generate_graph(smiles[i][j])
                if graph is None:
                    graphs=[]
                    break
                else:
                    graphs.append(graph['graph'])
            if len(graphs)!=0:
                if not self.combine_graph:
                    combined_graph=dgl.batch(graphs)
                else:
                    combined_graph=CombineGraph(graphs,add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                self.dataset.append([combined_graph,labels[i]])
    def load_smiles_dict_old(self,smiles_info,smiles_info_col):
        self.smiles=pd.read_csv(smiles_info,index_col=smiles_info_col[0]).to_dict()[smiles_info_col[1]]
    def load_smiles_dict(self,smiles_info,smiles_info_col):
        salt=pd.read_excel(smiles_info,sheet_name='Salt',index_col='Salt').to_dict()['SMILES']
        solvent=pd.read_excel(smiles_info,sheet_name='Molecule',index_col='Solvent').to_dict()['SMILES']
        new_solv={}
        for key in solvent.keys():
            new_solv[key.split(' (')[0]]=solvent[key]
        new_salt={}
        for key in salt.keys():
            new_salt[key.split(' (')[0]]=salt[key]
        self.smiles={**new_salt,**new_solv}
    def load_sample_dict_old(self,sample_info,start,end):
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
                    if self.binary:
                        combined_graph=dgl.batch([cation_graph['graph'],anion_graph['graph']])
                        # self.dataset.append([cation_graph['graph'],anion_graph['graph'],sub_group[i][2],sub_group[i][3],sub_group[i][4]])
                    else:
                        combined_graph=CombineGraph([cation_graph['graph'],anion_graph['graph']],add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                    self.dataset.append([combined_graph,sub_group[i][2],sub_group[i][3],sub_group[i][4]])
        if self.shuffle:
            random.shuffle(self.dataset)
    def load_sample_dict(self,sample_info,start,end):
        all_data=pd.read_excel(sample_info)[start:end]
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
                    graphs.append(graph['graph'])
            if len(graphs)!=0:
                if not self.combine_graph:
                    combined_graph=dgl.batch(graphs)
                else:
                    combined_graph=CombineGraph(graphs,add_global=True,bi_direction=True,add_self_loop=self.add_self_loop)
                self.dataset.append([combined_graph,np.concatenate([solv_mol[i],salt_mol[i]]).tolist(),ColumbicEfficiency[i]])
                # self.dataset.append([combined_graph,env_info[i],ColumbicEfficiency[i]])

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
    # def pretrained_featurizer(self,mol):
    #     can_feat=CanonicalAtomFeaturizer()
    #     self.pretrained_node_featurizer.eval()
    #     graph_data=mol_to_graph_data_obj_simple(mol)
    #     with torch.no_grad():
    #         node_emb=self.pretrained_node_featurizer(graph_data)
    #     con_emb=can_feat(mol)['h']
    #     return {'h':torch.cat([node_emb,con_emb],dim=-1)}

    def generate_graph(self,smiles):
        if self.featurize_edge:
            edge_featurizer=CanonicalBondFeaturizer()
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
        # if self.binary:
        #     return self.dataset[index][0],self.dataset[index][1],self.dataset[index][2:-1],self.dataset[index][-1]
        # else:
        sample={}
        sample['graph']=self.dataset[index][0]
        if self.add_feat:
            sample['add_features']=self.dataset[index][1:-1]
        else:
            sample['add_features']=[]
        return sample,self.dataset[index][-1]
        # return torch.Tensor(self.dataset[index][1]),torch.Tensor([self.dataset[index][2]])


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

def collate_solubility_binary(batch):
    samples=list(map(list,zip(*batch)))
    data={}
    data['cation']=dgl.batch(samples[0])
    data['anion']=dgl.batch(samples[1])
    data['add_features']=torch.Tensor(samples[2])
    return data,torch.Tensor(samples[3])

def collate_ce(batch):
    keys = list(batch[0][0].keys())[1:]
    samples=[]
    labels=[]
    for info in batch:
        samples.append(info[0].values())
        labels.append(info[1])
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['graphs'] = samples[0]
    for i,key in enumerate(keys):
        batched_sample[key] = torch.tensor(samples[i+1])
    return batched_sample,torch.Tensor(labels)
