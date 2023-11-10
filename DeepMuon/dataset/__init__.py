'''
Author: airscker
Date: 2022-09-20 20:03:40
LastEditors: airscker
LastEditTime: 2023-11-09 09:51:44
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''Basic dataset utils'''
from .SmilesGraphUtils.graph_operation import CombineGraph
from .SmilesGraphUtils.crystal_featurizer import (MPJCrystalGraphData, one_hot_encoding, one_hot_decoding,
                                                  mol_to_atom_bond_bigraph, smiles_to_atom_bond_graph)
from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from .SmilesGraphUtils.molecular_graph import (mol_to_graph,smiles_to_bigraph,mol_to_bigraph,
                                               smiles_to_complete_graph,mol_to_complete_graph,
                                               k_nearest_neighbors,mol_to_nearest_neighbor_graph,
                                               smiles_to_nearest_neighbor_graph)

'''Customized datasets'''
from .MinistData import MinistDataset
from .HailingData import HailingDataset_Direct2, HailingDataset_DirectV3
from .Pandax4TData import PandaxDataset
from .CMRData import NIIDecodeV2
from .XASData import ValenceDataset, ValenceDatasetV2
from .SolubilityData import (SmilesGraphData, MultiSmilesGraphData, collate_solubility,collate_solubility_binary,collate_ce)
from .XASDataV2 import XASSUMDataset, collate_XASSUM
from .AtomEmbedData import AtomMasking, MolSpaceDataset, collate_molspace, collate_atom_masking, collate_molspacev2
from .MolFoundation import MolFoundationDataset,collate_molfoundation,FoundationBasicDataset

__all__ = [
    'CombineGraph','MPJCrystalGraphData', 'one_hot_encoding', 'one_hot_decoding' ,
    'mol_to_atom_bond_bigraph', 'smiles_to_atom_bond_graph', 'CanonicalAtomFeaturizer','CanonicalBondFeaturizer',
    'mol_to_graph','smiles_to_bigraph','mol_to_bigraph','smiles_to_complete_graph',
    'mol_to_complete_graph','k_nearest_neighbors','mol_to_nearest_neighbor_graph','smiles_to_nearest_neighbor_graph',
    'PandaxDataset', 'HailingDataset_Direct2', 'HailingDataset_DirectV3',
    'NIIDecodeV2', 'ValenceDataset', 'ValenceDatasetV2', 'MinistDataset',
    'SmilesGraphData', 'MultiSmilesGraphData', 'collate_solubility','collate_solubility_binary','collate_ce', 'XASSUMDataset',
    'collate_XASSUM', 'AtomMasking', 'MolSpaceDataset', 'collate_molspace', 'collate_atom_masking', 'collate_molspacev2',
    'MolFoundationDataset','collate_molfoundation','FoundationBasicDataset'
]
