'''
Author: airscker
Date: 2023-09-05 18:38:28
LastEditors: airscker
LastEditTime: 2023-11-08 00:47:41
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch
import numpy as np

from typing import Union,Any
from itertools import combinations
from abc import ABCMeta,abstractmethod

from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

from .atom_feat_encoding import (atom_type_one_hot,atom_type_one_hot_alltable,atom_degree_one_hot,atom_implicit_valence_one_hot,
                                 atom_formal_charge,atom_num_radical_electrons,atom_hybridization_one_hot,atom_is_aromatic,atom_total_num_H_one_hot)
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Atom
from rdkit.Chem import AllChem,rdDistGeom,rdMolTransforms,AddHs
RDLogger.DisableLog('rdApp.*')

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
def atom_featurizer(atom):
    features=[]
    functions=[
        atom_type_one_hot,
        atom_degree_one_hot,
        atom_implicit_valence_one_hot,
        atom_formal_charge,
        atom_num_radical_electrons,
        atom_hybridization_one_hot,
        atom_is_aromatic,
        atom_total_num_H_one_hot,
    ]
    for func in functions:
        features.append(func(atom))
    return np.concatenate(features)

def mol_to_atom_bond_bigraph(mol:Chem.rdchem.Mol,add_Hs=False,only_atomic_num=False,return_bond_graph=True):
    '''
    ## Creat the graph of the molecule.

    ### Args:
        - mol: The molecule to be featurized.
        - add_Hs: If `True`, the hydrogens will be added to the molecule.
        - only_atomic_num: If `True`, the atom features are only the atomic number of the atom, otherwise, the atom features are the atom features encoded by `atom_featurizer`.
        - return_bond_graph: If `True`, the bond graph will be returned, otherwise, the bond graph will not be returned.

    ### Returns:
        - atom_graph: A `DGLGraph` object, the graph structure of the molecule
            - node features: `atomic_num`: The atomic number of the atom.
            - edge features: `bond_length`: The bond length between explicitly bonded atoms (Such as vad der waals or hydrogen bonds are excluded).
        - bond_graph: A `DGLGraph` object, the graph structure of the molecule
            - node features: 
                - `bond_length`: The bond length between explicitly bonded atoms. Which is the same as the `bond_length` in `atom_graph`, but their orders are different.
                - `bond_index`: The list of the bonded atoms' index (`e_ij`), which indicates the index of the atoms in `atom_graph`.
            - edge features:
                - `bond_angle`: The bond angles between connected bonds.
                - `angle_index`: The list of the connected bonds' index (`a_ijk = Angle(e_ij, e_jk)`), which indicates the index of the bonds in `bond_graph`.

    ### Example:

    >>> from rdkit import Chem
    >>> mol=Chem.MolFromSmiles('Br.C(COc1cccnc1)=NNC1=NCCN1')
    >>> atom_graph,bond_graph=mol_to_atom_edge_bigraph(mol, add_Hs=False)
    >>> atom_graph
        Graph(num_nodes=17, num_edges=34,
            ndata_schemes={'atomic_num': Scheme(shape=(), dtype=torch.int64)}
            edata_schemes={'bond_length': Scheme(shape=(), dtype=torch.float32)})
    >>> bond_graph
        Graph(num_nodes=34, num_edges=40,
            ndata_schemes={'bond_length': Scheme(shape=(), dtype=torch.float32), 'bond_index': Scheme(shape=(2,), dtype=torch.int64)}
            edata_schemes={'bond_angle': Scheme(shape=(), dtype=torch.float32), 'angle_index': Scheme(shape=(3,), dtype=torch.int64)})
    ```
    '''
    AllChem.EmbedMolecule(mol)
    if add_Hs:
        mol=AddHs(mol)
    adjacent_matrix=AllChem.GetAdjacencyMatrix(mol)
    bond_length_matrix=AllChem.Get3DDistanceMatrix(mol)
    conf=mol.GetConformer(0)

    src_list=[]
    dst_list=[]
    atomic_num=[]
    angle_index=[]
    bond_angle=[]
    for i in range(len(adjacent_matrix)):
        pos=np.where(adjacent_matrix[i]==1)[0]
        dst_pos=pos[pos>i]
        dst_list+=dst_pos.tolist()
        src_list+=[i]*len(dst_pos)
        if only_atomic_num:
            atomic_num.append(mol.GetAtomWithIdx(i).GetAtomicNum())
        else:
            # atomic_num.append(mol.GetAtomWithIdx(i).GetAtomicNum())
            atomic_num.append(atom_featurizer(mol.GetAtomWithIdx(i)))
        if len(pos)>1:
            _pos=np.array(list(combinations(pos,2)))
            _pos=np.insert(_pos,1,i,axis=1)
            angle_index+=_pos.tolist()
    src_dst=np.array([src_list,dst_list])
    src_dst=np.concatenate((src_dst,np.flip(src_dst,axis=0)),axis=1)

    for i in range(len(angle_index)):
        angle=rdMolTransforms.GetAngleDeg(conf,angle_index[i][0],angle_index[i][1],angle_index[i][2])
        bond_angle.append(angle)
    angle_index=np.array(angle_index).T
    angle_index=np.concatenate((angle_index,np.flip(angle_index,axis=0)),axis=1).T
    bond_angle=bond_angle+bond_angle
    bond_length=bond_length_matrix[src_dst[0],src_dst[1]]
    bond_index=src_dst.T

    atom_graph=dgl.graph([],idtype=torch.int64)
    atomic_num=np.array(atomic_num)
    atom_graph.add_nodes(len(atomic_num),
                        #  data={'atomic_num':torch.LongTensor(atomic_num)})
                        data={'atomic_num':torch.from_numpy(atomic_num).type(torch.float32)})
    atom_graph.add_edges(torch.LongTensor(src_dst[0]),
                         torch.LongTensor(src_dst[1]),
                         data={'bond_length':torch.FloatTensor(bond_length)})
    if not return_bond_graph:
        return atom_graph,None
    angle_vertex=[angle_index[:,0:2],angle_index[:,1:]]
    src_list=[]
    dst_list=[]
    for i in range(len(angle_index)):
        src_list.append(np.where(np.all(bond_index==angle_vertex[0][i],axis=1))[0].item())
        dst_list.append(np.where(np.all(bond_index==angle_vertex[1][i],axis=1))[0].item())

    bond_graph=dgl.graph([],idtype=torch.int64)
    bond_graph.add_nodes(len(bond_index),
                         data={'bond_length':torch.FloatTensor(bond_length)})
    bond_graph.add_edges(torch.LongTensor(src_list),
                         torch.LongTensor(dst_list),
                         data={'bond_angle':torch.FloatTensor(bond_angle)})
    bond_graph.ndata['bond_index']=torch.LongTensor(bond_index)
    bond_graph.edata['angle_index']=torch.LongTensor(angle_index)

    return atom_graph,bond_graph

def smiles_to_atom_bond_graph(smiles:str,add_Hs=False,only_atomic_num=False,return_bond_graph=True):
    '''
    ## Creat the graph of the molecule.

    ### Args:
        - smiles: The molecule to be featurized, given by SMILES.
        - add_Hs: If `True`, the hydrogens will be added to the molecule.
        - only_atomic_num: If `True`, the atom features are only the atomic number of the atom, otherwise, the atom features are the atom features encoded by `atom_featurizer`.
        - return_bond_graph: If `True`, the bond graph will be returned, otherwise, the bond graph will not be returned.

    ### Returns:
        - atom_graph: A `DGLGraph` object, the graph structure of the molecule
            - node features: `atomic_num`: The atomic number of the atom.
            - edge features: `bond_length`: The bond length between explicitly bonded atoms (Such as vad der waals or hydrogen bonds are excluded).
        - bond_graph: A `DGLGraph` object, the graph structure of the molecule
            - node features: 
                - `bond_length`: The bond length between explicitly bonded atoms. Which is the same as the `bond_length` in `atom_graph`, but their orders are different.
                - `bond_index`: The list of the bonded atoms' index (`e_ij`), which indicates the index of the atoms in `atom_graph`.
            - edge features:
                - `bond_angle`: The bond angles between connected bonds.
                - `angle_index`: The list of the connected bonds' index (`a_ijk = Angle(e_ij, e_jk)`), which indicates the index of the bonds in `bond_graph`.

    ### Example:

    >>> atom_graph,bond_graph=smiles_to_atom_edge_bigraph(smiles='Br.C(COc1cccnc1)=NNC1=NCCN1, add_Hs=False)
    >>> atom_graph
        Graph(num_nodes=17, num_edges=34,
            ndata_schemes={'atomic_num': Scheme(shape=(), dtype=torch.int64)}
            edata_schemes={'bond_length': Scheme(shape=(), dtype=torch.float32)})
    >>> bond_graph
        Graph(num_nodes=34, num_edges=40,
            ndata_schemes={'bond_length': Scheme(shape=(), dtype=torch.float32), 'bond_index': Scheme(shape=(2,), dtype=torch.int64)}
            edata_schemes={'bond_angle': Scheme(shape=(), dtype=torch.float32), 'angle_index': Scheme(shape=(3,), dtype=torch.int64)})
    ```
    '''
    try:
        mol=Chem.MolFromSmiles(smiles)
    except:
        return None,None
    if mol is None:
        return None,None
    try:
        atom_graph,bond_graph=mol_to_atom_bond_bigraph(mol,add_Hs,only_atomic_num,return_bond_graph)
    except:
        return None,None
    return atom_graph,bond_graph
