'''
Author: airscker
Date: 2022-09-20 20:03:40
LastEditors: airscker
LastEditTime: 2024-07-03 01:17:27
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''Basic dataset utils'''
from .SmilesGraphUtils.graph_operation import CombineGraph
from .SmilesGraphUtils.crystal_featurizer import (MPJCrystalGraphData, energy_level_ev, one_hot_encoding, one_hot_decoding,
                                                  mol_to_atom_bond_bigraph, smiles_to_atom_bond_graph, ENERGY_LEVEL_UNIT_EV, energy_gaps)
from .SmilesGraphUtils.atom_feat_encoding import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from .SmilesGraphUtils.molecular_graph import (mol_to_graph,smiles_to_bigraph,mol_to_bigraph,
                                               smiles_to_complete_graph,mol_to_complete_graph,
                                               k_nearest_neighbors,mol_to_nearest_neighbor_graph,
                                               smiles_to_nearest_neighbor_graph)

'''Customized datasets'''
from .MinistData import MinistDataset
from .hailing import HailingDataset_Direct2, HailingDataset_DirectV3, PandaxDataset
from .CMRData import NIIDecodeV2
from .XASData import ValenceDataset, ValenceDatasetV2
from .SolubilityData import (SmilesGraphData, MultiSmilesGraphData, collate_solubility,collate_solubility_binary,collate_ce)
from .XASDataV2 import XASSUMDataset, collate_XASSUM
from .XASDataV3 import XASSUMDatasetV3, collate_xas_struc, collate_xas_atom, alphaxasdataset, collate_alphaxas
from .AtomEmbedData import AtomMasking, MolSpaceDataset, collate_molspace, collate_atom_masking, collate_molspacev2
from .MolFoundation import MolFoundationDataset,collate_molfoundation,FoundationBasicDataset

__all__ = [
    'CombineGraph', 'MPJCrystalGraphData', 'energy_level_ev', 'one_hot_encoding',
    'one_hot_decoding', 'mol_to_atom_bond_bigraph','ENERGY_LEVEL_UNIT_EV', 'energy_gaps',
    'smiles_to_atom_bond_graph', 'CanonicalAtomFeaturizer',
    'CanonicalBondFeaturizer', 'mol_to_graph', 'smiles_to_bigraph',
    'mol_to_bigraph', 'smiles_to_complete_graph', 'mol_to_complete_graph',
    'k_nearest_neighbors', 'mol_to_nearest_neighbor_graph',
    'smiles_to_nearest_neighbor_graph', 'PandaxDataset',
    'HailingDataset_Direct2', 'HailingDataset_DirectV3', 'NIIDecodeV2',
    'ValenceDataset', 'ValenceDatasetV2', 'MinistDataset', 'SmilesGraphData',
    'MultiSmilesGraphData', 'collate_solubility', 'collate_solubility_binary',
    'collate_ce', 'XASSUMDataset', 'collate_XASSUM', 'AtomMasking',
    'MolSpaceDataset', 'collate_molspace', 'collate_atom_masking',
    'collate_molspacev2', 'MolFoundationDataset', 'collate_molfoundation',
    'FoundationBasicDataset', 'XASSUMDatasetV3', 'collate_xas_struc',
    'collate_xas_atom', 'alphaxasdataset', 'collate_alphaxas'
]
