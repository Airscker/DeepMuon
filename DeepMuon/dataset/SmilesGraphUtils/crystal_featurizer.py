'''
Author: airscker
Date: 2023-09-05 18:38:28
LastEditors: airscker
LastEditTime: 2023-10-08 12:23:42
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

def one_hot_decoding(data:Any):
    if isinstance(data,np.ndarray):
        return np.argmax(data)+1
    elif isinstance(data,torch.Tensor):
        return torch.argmax(data)+1
    elif isinstance(data,list):
        return data.index(1)+1

def one_hot_encoding(data:int, code_length:int, encode=True):
    if encode:
        code=np.zeros(code_length)
        code[data-1]=1
        return code
    else:
        return data

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

    ### Features may included:
        - node features:
            - One hot encoding of the atom type. The supported atom types include all atoms in the periodic table, which includes 118 types of atoms.
            - Every atom's 3d coordinate in the crystal.
            - Every atoms's mass (dim-1, unavailable for neighbor graph).
            - Every atom's degree (dim-1, unavailable for neighbor graph).
        - edge features:
            - Every edge's image (dim-3, only available for neighbor graph).
            - Every edge's length (dim-1, only available for neighbor graph).

    
    ### Args:
        - structure: The crystal structure to be featurized, if `None`, the graph will be created later using `creat_graph` function.
        - bidirectional: If `True`, the graph is bidirectional.
        - self_loop: If `True`, the graph has self loop.
        - onehot_encode: If `True`, the atom type and atom degree are one hot encoded.
        - neighbor_graph: If `True`, the graph is a neighborhood graph (atoms within neigborhood are all connected), otherwise, the graph is a bond graph (only actual bonds exist in th graph).
        - atom_neigh_cutoff: The cutoff radius of the neighborhood graph, only available when `neighbor_graph` is `True`, the UNIT is `A`.
    
    ### Returns:
        - graph: 
            - neighbor_graph=False:
                A `DGLGraph` object, the graph structure of the crystal structure, which contains only node features, namely `graph.ndata['feat']`\n
                If `bidirectional` is `True`, the graph is bidirectional, otherwise, the graph is unidirectional.\n
                If `self_loop` is `True`, the graph has self loop, otherwise, the graph has no self loop.\n
                If `onehot_encode` is `True`, the atom type and atom degree are one hot encoded, otherwise, the atom type and atom degree are not one hot encoded.
                And node features' dimensions are `123` when `onehot_encode` is `True`, otherwise, the node features' dimensions are `6`.
            - neighbor_graph=True:
                A `DGLGraph` object, the graph structure of the crystal structure, which contains node features `graph.ndata['feat']` and edge features `graph.edata['feat']`\n
                Arg `bidirectional` is unapplicable because we consider neighborhood graph here.\n
                If `self_loop` is `True`, the graph has self loop, otherwise, the graph has no self loop.\n
                If `onehot_encode` is `True`, the atom type and atom degree are one hot encoded, otherwise, the atom type and atom degree are not one hot encoded.
                And node features' dimensions are `123` when `onehot_encode` is `True`, otherwise, the node features' dimensions are `4`.
                Also edge features' dimensions is `4`.
    """

    def __init__(self,structure=None,bidirectional:bool=True,self_loop=False,onehot_encode:bool=False,neighbor_graph=False,atom_neigh_cutoff=5) -> None:
        super().__init__()
        self.bidirectional=bidirectional
        self.self_loop=self_loop
        # self.dim_degree=dim_degree
        self.onehot_encode=onehot_encode
        self.neighbor_graph=neighbor_graph
        self.atom_neigh_cutoff=atom_neigh_cutoff
        self.CrystalNN=CrystalNN()
        if structure is not None:
            self.creat_graph(structure)
    def creat_graph(self,*args,**kwargs):
        if self.neighbor_graph:
            return self.neigh_graph(*args,**kwargs)
        else:
            return self.bond_graph(*args,**kwargs)
    def neigh_graph(self,structure:Structure=None):
        n_atoms = len(structure)
        atomic_number = torch.tensor([one_hot_encoding(site.specie.Z,118,self.onehot_encode) for site in structure])
        atom_coord = torch.tensor(structure.cart_coords)
        # lattice = structure.lattice.matrix
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(r=self.atom_neigh_cutoff, sites=structure.sites, numerical_tol=1e-8)
        graph=dgl.graph([],idtype=torch.int64)
        graph.add_nodes(n_atoms,
                        data={'feat':torch.cat([atomic_number,atom_coord],dim=-1)})
        graph.add_edges(center_index,
                        neighbor_index,
                        data={'feat':torch.cat([torch.tensor(image),torch.tensor(distance)],dim=-1)})
        self.graph=graph
        return graph
    def bond_graph(self,structure:Structure=None,adj_matrix:dict=None):
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