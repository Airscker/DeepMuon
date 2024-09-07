'''
Author: airscker
Date: 2024-02-01 13:59:34
LastEditors: airscker
LastEditTime: 2024-07-04 01:33:34
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import random
import pickle as pkl
import numpy as np

import dgl
import torch

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset
from .SmilesGraphUtils.crystal_featurizer import MPJCrystalGraphData

periodic_table=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
                'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


class XASSUMDatasetV3(Dataset):

    def __init__(self,
                 data_ann: str = None,
                 xas_type: str = 'XANES',
                 xas_edge: str = 'K',
                 convert_graph: bool = True,
                 self_loop: bool = False,
                 onehot_encode: bool = True,
                 cutoff: float = 6.0,
                 shuffle: bool = True,
                 verbose=True) -> None:
        super().__init__()
        '''
        ## Load the XAS data and the corresponding structure data

        ### Args:
            - data_ann: str, the path of the data annotation file, each line is the path of the data file.
            - xas_type: str, the type of the XAS data, 'XANES' or 'EXAFS' or 'XAFS'.
            - xas_edge: str, the edge of the XAS data, 'K' or 'L3' or 'L2' or 'L2,3' or 'M'.
            - convert_graph: bool, whether to convert the crystal graph to DGL graph.
            - self_loop: bool, whether to add self loop to the graph nodes.
            - onehot_encode: bool, whether to onehot encode the atom type.
            - cutoff: float, the cutoff distance for the atom neighbor graph.
            - shuffle: bool, whether to shuffle the data.
            - verbose: bool, whether to show the loading bar.
        
        ### Returns:
            - A list of data, each element of which is a list of three elements: `[graph,struc_prompt,spectrum,absorbing_atom]`.
                - graph: The crystal graph of the material.
                - struc_prompt: The features of the structure.
                - spectrum: The absorbing spectrum of the corresponding atom.
                - absorbing_atom: The type of the absorbing atom.
        '''
        self.xas_type = xas_type
        self.xas_edge = xas_edge
        self.shuffle = shuffle
        self.verbose = verbose
        self.convert_graph = convert_graph
        self.onehot_encode = onehot_encode
        self.xas_length = {'XANES': 100, 'EXAFS': 500, 'XAFS': 600}
        self.edge_set = {'K': 0, 'L3': 1, 'L2': 2, 'L2,3': 3, 'M': 4}
        self.graph_featurizer = MPJCrystalGraphData(
            self_loop=self_loop,
            onehot_encode=onehot_encode,
            atom_neigh_cutoff=cutoff)
        self._load_data(data_ann)

    def _load_data(self, data_ann):
        with open(data_ann, 'r') as f:
            self.data_ann = f.readlines()
        if self.shuffle:
            random.shuffle(self.data_ann)
        self.xas_set = []
        self.xas_struc_map = {}
        self.struc_set = []
        if self.verbose:
            bar = tqdm(range(len(self.data_ann)), desc='Loading Data: ')
        else:
            bar = range(len(self.data_ann))
        for i in range(len(self.data_ann)):
            self.data_ann[i] = self.data_ann[i].split('\n')[0]
            with open(self.data_ann[i], 'rb') as f:
                data = pkl.load(f)
            self.struc_set.append([
                self.graph_featurizer(data[0]['structure'])
                if self.convert_graph else None, data[2]
            ])
            spectrums = data[1]
            for j in range(len(spectrums)):
                spec_type = str(spectrums[j]['spectrum_type'])
                spec_edge = str(spectrums[j]['edge'])
                spec_atom = periodic_table.index(
                    str(spectrums[j]['absorbing_element']))
                if spec_type == self.xas_type and spec_edge == self.xas_edge:
                    spec_data = self._spec_len_check(spectrums[j]['spectrum'])
                    if spec_data is not None:
                        self.xas_set.append([spec_data, spec_atom])
                        self.xas_struc_map[len(self.xas_set) - 1] = i

    def _spec_len_check(self, spectrum):
        spec = [spectrum.x, spectrum.y]
        if len(spec[0]) != self.xas_length[self.xas_type]:
            return None
        else:
            return np.array(spec)

    def __len__(self) -> int:
        return len(self.xas_set)

    def __getitem__(self, index):
        graph, struc_prompt = self.struc_set[self.xas_struc_map[index]]
        spec_data, spec_atom = self.xas_set[index]
        # if self.onehot_encode:
        #     _atomic_num = torch.argmax(graph.ndata['atomic_num'], dim=-1)
        # else:
        #     _atomic_num = graph.ndata['atomic_num']
        # _abs_mask = torch.zeros(graph.num_nodes())
        # _abs_mask[_atomic_num == spec_atom] = 1
        # graph.ndata['abs_mask'] = _abs_mask
        # graph.ndata['spec_x']=torch.from_numpy(spec_data[0]).repeat(graph.num_nodes(),1)
        return graph, struc_prompt, spec_data, spec_atom

class alphaxasdataset(Dataset):

    def __init__(self,
                 data_ann: str = None,
                 xas_type: str = 'XANES',
                 xas_edge: str = 'K',
                 cutoff: float = 6.0,
                 shuffle: bool = True,
                 verbose=True,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        '''
        ## Load the XAS data and the corresponding structure data

        ### Args:
            - data_ann: str, the path of the data annotation file, each line is the path of the data file.
            - xas_type: str, the type of the XAS data, 'XANES' or 'EXAFS' or 'XAFS'.
            - xas_edge: str, the edge of the XAS data, 'K' or 'L3' or 'L2' or 'L2,3' or 'M'.
            - cutoff: float, the cutoff distance for the atom neighbor graph.
            - shuffle: bool, whether to shuffle the data.
            - verbose: bool, whether to show the loading bar.
        
        ### Returns:
            - A list of data, each element of which is a list of three elements: `[graph,struc_prompt,spectrum,absorbing_atom]`.
                - graph: The crystal graph of the material.
                - struc_prompt: The features of the structure.
                - spectrum: The absorbing spectrum of the corresponding atom.
                - absorbing_atom: The type of the absorbing atom.
        '''
        self.xas_type = xas_type
        self.xas_edge = xas_edge
        self.shuffle = shuffle
        self.verbose = verbose
        self.xas_length = {'XANES': 100, 'EXAFS': 500, 'XAFS': 600}
        self.edge_set = {'K': 0, 'L3': 1, 'L2': 2, 'L2,3': 3, 'M': 4}
        self.graph_featurizer = MPJCrystalGraphData(
            self_loop=False,
            onehot_encode=False,
            atom_neigh_cutoff=cutoff)
        self._load_data(data_ann)

    def _load_data(self, data_ann):
        with open(data_ann, 'r') as f:
            self.data_ann = f.readlines()
        if self.shuffle:
            random.shuffle(self.data_ann)
        self.xas_set = []
        self.xas_struc_map = {}
        self.struc_set = []
        if self.verbose:
            bar = tqdm(range(len(self.data_ann)), desc='Loading Data: ')
        else:
            bar = range(len(self.data_ann))
        for i in bar:
            self.data_ann[i] = self.data_ann[i].split('\n')[0]
            with open(self.data_ann[i], 'rb') as f:
                data = pkl.load(f)
            self.struc_set.append([
                # self.graph_featurizer(data[0]['structure'])
                # if self.convert_graph else None, data[2]
                *self.graph_featurizer.molecule_tensor(data[0]['structure'])
            ])
            spectrums = data[1]
            for j in range(len(spectrums)):
                spec_type = str(spectrums[j]['spectrum_type'])
                spec_edge = str(spectrums[j]['edge'])
                spec_atom = periodic_table.index(
                    str(spectrums[j]['absorbing_element']))
                if spec_type == self.xas_type and spec_edge == self.xas_edge:
                    spec_data = self._spec_len_check(spectrums[j]['spectrum'])
                    if spec_data is not None:
                        self.xas_set.append([spec_data, spec_atom])
                        self.xas_struc_map[len(self.xas_set) - 1] = i

    def _spec_len_check(self, spectrum):
        spec = [spectrum.x, spectrum.y]
        if len(spec[0]) != self.xas_length[self.xas_type]:
            return None
        else:
            return np.array(spec)

    def __len__(self) -> int:
        return len(self.xas_set)

    def __getitem__(self, index):
        molecular_matrix, atom_feature = self.struc_set[self.xas_struc_map[index]]
        spec_data, spec_atom = self.xas_set[index]
        return molecular_matrix, atom_feature, len(atom_feature), spec_data, spec_atom

def feature_padding(molecular_matrix,atom_feature,length_list):
    max_len=max(length_list)
    molecular_matrix_masks=torch.zeros(len(length_list),max_len,max_len)
    atom_feature_masks=torch.zeros(len(length_list),max_len)
    for i in range(len(molecular_matrix)):
        molecular_matrix[i]=F.pad(molecular_matrix[i],(0,0,0,max_len-length_list[i],0,max_len-length_list[i]),value=0)
        atom_feature[i]=F.pad(atom_feature[i],(0,max_len-length_list[i]),value=0)
        molecular_matrix_masks[i,:length_list[i],:length_list[i]]=1
        atom_feature_masks[i,:length_list[i]]=1
    return torch.stack(molecular_matrix),torch.stack(atom_feature),molecular_matrix_masks.bool(),atom_feature_masks.bool()

def collate_alphaxas(batch):
    if len(batch)==1:
        samples=batch[0]
        molecular_matrix,atom_feature,length_list=samples[0],samples[1],samples[2]
        spectrum=torch.Tensor(samples[3]).float().unsqueeze(0)
        molecular_matrix=molecular_matrix.unsqueeze(0)
        atom_feature=atom_feature.unsqueeze(0)
        atom_feature_masks=None
        molecular_matrix_masks=None
    else:
        samples=list(map(list, zip(*batch)))
        molecular_matrix,atom_feature,length_list=samples[0],samples[1],samples[2]
        spectrum=torch.Tensor(samples[3]).float()
        molecular_matrix,atom_feature,molecular_matrix_masks,atom_feature_masks=feature_padding(molecular_matrix,atom_feature,length_list)
    # atoms=torch.Tensor(samples[4]).long()
    # print(spectrum.shape,molecular_matrix.shape,atom_feature.shape)
    # try:
    #     print(molecular_matrix_masks.shape,atom_feature_masks.shape)
    # except:
    #     pass
    return [spectrum[:,0,:]/1000,atom_feature,molecular_matrix,atom_feature_masks,molecular_matrix_masks],spectrum

def collate_xas_struc(batch):
    samples = list(map(list, zip(*batch)))
    samples[0]=dgl.batch(samples[0])
    spectrum=torch.Tensor(samples[2])
    samples[2]=spectrum
    samples[3]=torch.Tensor(samples[3])
    return samples,spectrum

def collate_xas_atom(batch):
    samples=list(map(list, zip(*batch)))
    spectrum = torch.Tensor(samples[2]).float()
    atoms=torch.Tensor(samples[3]).long()
    return spectrum,atoms
