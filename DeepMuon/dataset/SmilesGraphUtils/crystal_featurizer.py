'''
Author: airscker
Date: 2023-09-05 18:38:28
LastEditors: airscker
LastEditTime: 2023-09-16 00:41:32
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

from typing import Any
import dgl
import torch
import numpy as np

from abc import ABCMeta,abstractmethod
from typing import Union
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from rdkit.Chem.rdchem import Atom

def one_hot_encoding(data:int, code_length:int):
    code=np.zeros(code_length)
    code[data]=1
    return code

class BaseCrystalGraphData(object,metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    def _sort_dict_key(self,data:dict):
        new_data={}
        for key in sorted(data.keys()):
            new_data[key]=data[key]
        return new_data
    def _get_atom_index(self,atom:str):
        atom=Atom(atom)
        return atom.GetAtomicNum()
    def _get_atom_mass(self,atom:str):
        atom=Atom(atom)
        return atom.GetMass()
    @property
    def num_atoms(self):
        return 118
    @abstractmethod
    def creat_graph(self):
        pass
    
    def __call__(self, *args: Any, **kwds: Any):
        return self.creat_graph(*args, **kwds)

class MPJCrystalGraphData(BaseCrystalGraphData):
    """
    ## Crystal atom graph featurizer for crystal structures from Materials Project (https://next-gen.materialsproject.org/).

    ### The atom features may include:
        - One hot encoding of the atom type. The supported atom types include all atoms in the periodic table, which includes 118 types of atoms.
        - Every atom's 3d coordinate in the crystal.
        - Every atoms's mass.
        - Every atom's degree.
    
    ### Args:
        - structure: The crystal structure to be featurized, if `None`, the graph will be created later using `creat_graph` function.
        - bidirectional: If `True`, the graph is bidirectional.
        - self_loop: If `True`, the graph has self loop.
        - onehot_encode: If `True`, the atom type and atom degree are one hot encoded.
    
    ### Returns:
        - graph: A `DGLGraph` object, the graph structure of the crystal structure, which contains only node features, namely `graph.ndata['feat']`\n
            If `bidirectional` is `True`, the graph is bidirectional, otherwise, the graph is unidirectional.\n
            If `self_loop` is `True`, the graph has self loop, otherwise, the graph has no self loop.\n
            If `onehot_encode` is `True`, the atom type and atom degree are one hot encoded, otherwise, the atom type and atom degree are not one hot encoded.
            And node features' dimensions of are `123` when `onehot_encode` is `True`, otherwise, the node features' dimensions are `6`.
    """

    def __init__(self,structure=None,bidirectional:bool=True,self_loop=False,onehot_encode:bool=False) -> None:
        super().__init__()
        self.bidirectional=bidirectional
        self.self_loop=self_loop
        # self.dim_degree=dim_degree
        self.onehot_encode=onehot_encode
        self.CrystalNN=CrystalNN()
        if structure is not None:
            self.creat_graph(structure)
    def creat_graph(self,structure:Structure=None,adj_matrix:dict=None):
        '''
        ## Creat the graph of the crystal structure.

        ### Args:
            - structure: The crystal structure to be featurized.
            - adj_matrix: The adjacency matrix of the crystal structure, which can be obtained by `CrystalNN.get_bonded_structure(structure)`.
        
        ### Tips:
            - If `adj_matrix` is `None`, the `structure` parameter will be omitted. This operation was designed to reduce the time cost of loading data.
        '''
        if adj_matrix is None:
            adj_matrix=self.CrystalNN.get_bonded_structure(structure).as_dict()
        else:
            adj_matrix=adj_matrix
        struc_graph_nodes=adj_matrix['graphs']['nodes']
        struc_graph_adj=adj_matrix['graphs']['adjacency']
        node_info=[]
        for i in range(len(struc_graph_nodes)):
            atom=struc_graph_nodes[i]['specie']
            atom_num=self._get_atom_index(atom)
            atom_num=one_hot_encoding(atom_num,self.num_atoms) if self.onehot_encode else [atom_num]
            atom_mass=self._get_atom_mass(atom)
            atom_coord=struc_graph_nodes[i]['coords']
            node_info.append([*atom_coord,*atom_num,atom_mass])
        node_index=[]
        for i in range(len(struc_graph_adj)):
            neighbors=[]
            for item in struc_graph_adj[i]:
                neighbors.append(item['id'])
                node_index.append([i,item['id']])
                if self.bidirectional:
                    node_index.append([item['id'],i])
            if self.self_loop:
                node_index.append([i,i])
            # atom_degree=one_hot_encoding(len(neighbors),self.dim_degree) if self.onehot_encode else [len(neighbors)]
            atom_degree=len(neighbors)
            node_info[i]=[*node_info[i],atom_degree]
        node_index=list(zip(*node_index))
        graph=dgl.graph((node_index[0],node_index[1]),num_nodes=len(struc_graph_nodes))
        graph.ndata['feat']=torch.Tensor(node_info).type(torch.float32)
        self.graph=graph
        return graph