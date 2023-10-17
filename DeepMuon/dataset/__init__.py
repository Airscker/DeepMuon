'''
Author: airscker
Date: 2022-09-20 20:03:40
LastEditors: airscker
LastEditTime: 2023-10-15 16:29:27
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''Basic dataset utils'''
from .SmilesGraphUtils.crystal_featurizer import MPJCrystalGraphData, one_hot_encoding, one_hot_decoding
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
from .SolubilityData import SmilesGraphData, MultiSmilesGraphData, collate_solubility, mol_to_graph_data_obj_simple, PreTrainedNodeEmbedding
from .XASDataV2 import XASSUMDataset, collate_XASSUM
from .AtomEmbedData import AtomMasking, collate_atom_masking

__all__ = [
    'MPJCrystalGraphData', 'one_hot_encoding', 'one_hot_decoding' ,
    'CanonicalAtomFeaturizer','CanonicalBondFeaturizer',
    'mol_to_graph','smiles_to_bigraph','mol_to_bigraph','smiles_to_complete_graph',
    'mol_to_complete_graph','k_nearest_neighbors','mol_to_nearest_neighbor_graph','smiles_to_nearest_neighbor_graph',
    'PandaxDataset', 'HailingDataset_Direct2', 'HailingDataset_DirectV3',
    'NIIDecodeV2', 'ValenceDataset', 'ValenceDatasetV2', 'MinistDataset',
    'SmilesGraphData', 'MultiSmilesGraphData', 'collate_solubility',
    'mol_to_graph_data_obj_simple', 'PreTrainedNodeEmbedding', 'XASSUMDataset',
    'collate_XASSUM', 'AtomMasking', 'collate_atom_masking',
]
