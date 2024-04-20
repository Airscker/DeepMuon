'''
Author: airscker
Date: 2022-09-20 19:32:21
LastEditors: airscker
LastEditTime: 2024-04-19 21:47:56
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

'''Basic block modules for constructing models'''
from .base import *

'''Models of project Hailing'''
from .hailing import *

'''Models of project CMR'''
from .cmr import *

'''Other models'''
from .Minist import MinistModel
from .MLPMixer import MLPMixer, XASMLP
from .MINES import SolvGNN, SolvGNNV2, SolvGNNV3, SolvGNNV4, SolvGNNV5, SolvLinear, SolvGNNV6, SolvGNNV7, AdjGNN
from .CrystalXAS import GINConv, CrystalXASV1, CrystalXASV2, CrystalXASV3, CrystalXASV4,XASStructure,XAS_Atom,XASStructureV2
from .MolPT import AtomEmbedding
from .MolDS import MolSpaceMultiHeadAttention, MolSpaceTransformer, MolSpaceGNNFeaturizer, MolSpace
from .MolGNN import MolSpaceGNN, MulMolSpace, MolProperty, TestGNN

# __all__ = [
#     'MLP3', 'MinistModel', 'ResMax2', 'ResMax3', 'VST',
#     'screening_model', 'fusion_model', 'Dense4012FrameRNN', 'EGFR_SwinT',
#     'EGFR_SwinTV2', 'MLPMixer', 'XASMLP', 'SolvGNN', 'SolvGNNV2', 'SolvGNNV3',
#     'SolvGNNV4', 'SolvGNNV5', 'SolvLinear', 'SolvGNNV6', 'SolvGNNV7',
#     'GINConv', 'CrystalXASV1', 'XASStructure', 'XAS_Atom',
#     'CrystalXASV2', 'CrystalXASV3', 'CrystalXASV4', 'AtomEmbedding',
#     'MolSpaceMultiHeadAttention', 'MolSpaceTransformer',
#     'MolSpaceGNNFeaturizer', 'MolSpace', 'MolSpaceGNN', 'MulMolSpace',
#     'MolProperty', 'TestGNN', 'AdjGNN'
# ]
