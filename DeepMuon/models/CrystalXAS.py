'''
Author: airscker
Date: 2023-09-10 17:32:44
LastEditors: airscker
LastEditTime: 2023-09-18 12:41:09
Description: NULL

Copyright (C) 2023 by Deep Graph Library, All Rights Reserved. 
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import torch
import dgl
import torch.nn.functional as F
from torch import nn
from .base import MLPBlock
from dgl.nn.pytorch import GINConv
from typing import Union


class CrystalXASV1(nn.Module):
    '''
    ## CrystalXAS model for XAS spectrum prediction.

    ### Args:
        - gnn_hidden_dims: The hidden dimensions of the GNN layers, the depth of GNN part is `len(gnn_hidden_dims)-1`.
        - feat_dim: The dimension of the atom features.
        - prompt_dim: The dimension of the prompt features.
        - mlp_hidden_dims: The hidden dimensions of the MLP layers, the depth of MLP part is `len(mlp_hidden_dims)-1`.
        - mlp_dropout: The dropout rate of the MLP layers.
        - xas_type: The type of XAS data to be loaded. The supported types include `XANES`, `EXAFS` and `XAFS`.
    '''

    def __init__(self,
                 gnn_hidden_dims: Union[list,int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__()
        xas_types = ['XANES', 'EXAFS', 'XAFS']
        assert xas_type in xas_types, f"'xas_type' must be in {xas_types}, but {xas_type} was given."
        self.xas_type = xas_type
        if isinstance(gnn_hidden_dims, int):
            gnn_hidden_dims = [gnn_hidden_dims] * gnn_layers
            self.gnn_res_connection = gnn_res_connection
        else:
            if gnn_res_connection:
                print('Residual connection is unavailable when `gnn_hidden_dims` is a list. That is, dimensions of GNN layers must be the same.')
            self.gnn_res_connection = False
        self.XANES = MLPBlock(gnn_hidden_dims[-1] + prompt_dim,
                              100,
                              mlp_hidden_dims,
                              mode='NAD',
                              activation=nn.ReLU,
                              normalization=nn.BatchNorm1d,
                              dropout_rate=mlp_dropout,
                              bias=True)
        self.EXAFS = MLPBlock(gnn_hidden_dims[-1] + prompt_dim,
                              500,
                              mlp_hidden_dims,
                              mode='NAD',
                              activation=nn.ReLU,
                              normalization=nn.BatchNorm1d,
                              dropout_rate=mlp_dropout,
                              bias=True)
        # self.xas_generator = nn.Sequential(
        #     ResidualUnit(1,1,16,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     ResidualUnit(1,16,32,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     ResidualUnit(1,32,2,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     )
        self.prompt_nn = nn.Linear(prompt_dim, prompt_dim)
        gnn_hidden_dims = [feat_dim] + gnn_hidden_dims
        GNN_Transform = nn.ModuleList([
            nn.Linear(gnn_hidden_dims[i], gnn_hidden_dims[i + 1])
            for i in range(len(gnn_hidden_dims) - 1)
        ])
        self.GIN = nn.ModuleList([
            GINConv(apply_func=GNN_Transform[i],
                    aggregator_type='sum',
                    init_eps=0,
                    learn_eps=False,
                    activation=F.relu)
            for i in range(len(gnn_hidden_dims) - 1)
        ])

    def forward(self, data, device):
        graph = data['graph'].to(device)
        prompt = data['prompt'].to(device)
        prompt = self.prompt_nn(prompt)
        atom_features = graph.ndata['feat']
        with graph.local_scope():
            for i in range(len(self.GIN)):
                result = self.GIN[i](graph, atom_features)
                if self.gnn_res_connection and i != 0:
                    atom_features = result + atom_features
                else:
                    atom_features = result
            graph.ndata['feat'] = atom_features
            atom_features = dgl.sum_nodes(graph, 'feat')
        if self.xas_type == 'XANES':
            spectrum = self.XANES(torch.cat([atom_features, prompt], dim=-1))
        elif self.xas_type == 'EXAFS':
            spectrum = self.EXAFS(torch.cat([atom_features, prompt], dim=-1))
        else:
            spectrum = torch.cat([self.XANES(torch.cat([atom_features, prompt], dim=-1)),
                                  self.EXAFS(torch.cat([atom_features, prompt], dim=-1))],
                                  dim=-1)
        # spectrum = self.xas_generator(spectrum.unsqueeze(1))
        return spectrum

class CrystalXASV2(nn.Module):
    '''
    ## CrystalXAS model for XAS spectrum prediction.

    ### Args:
        - gnn_hidden_dims: The hidden dimensions of the GNN layers, the depth of GNN part is `len(gnn_hidden_dims)-1`.
            If integer is given, the hidden dimensions of all GNN layers will be the same.
        - gnn_layers: The number of GNN layers, only usable when `gnn_hidden_dims` is interger.
        - gnn_res_connection: Whether to use residual connection in GNN part.
        - feat_dim: The dimension of the atom features.
        - prompt_dim: The dimension of the prompt features.
        - prompt_hidden_dim: The output dimension of the prompt NN.
        - normnn_dim: The output dimension of the normalization NN.
        - mlp_hidden_dims: The hidden dimensions of the MLP layers, the depth of MLP part is `len(mlp_hidden_dims)-1`.
        - mlp_dropout: The dropout rate of the MLP layers.
        - xas_type: The type of XAS data to be loaded. The supported types include `XANES`, `EXAFS` (NO `XAFS`).
    '''

    def __init__(self,
                 gnn_hidden_dims: Union[list,int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int=1024,
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__()
        xas_types = {'XANES':100, 'EXAFS':500}
        assert xas_type in xas_types.keys(), f"'xas_type' must be in {xas_types.keys()}, but {xas_type} was given."
        self.xas_type = xas_type
        if isinstance(gnn_hidden_dims, int):
            self.gnn_hidden_dims = [gnn_hidden_dims] * gnn_layers
            self.gnn_res_connection = gnn_res_connection
        else:
            if gnn_res_connection:
                print('Residual connection is unavailable when `gnn_hidden_dims` is a list. That is, dimensions of GNN layers must be the same.')
            self.gnn_res_connection = False
            self.gnn_hidden_dims = gnn_hidden_dims
        self.gnn_hidden_dims = [feat_dim] + self.gnn_hidden_dims
        self.NormalizeNN=nn.Sequential(
            MLPBlock(dim_input=self.gnn_hidden_dims[-1],
                     dim_output=normnn_dim,
                     hidden_sizes=mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True),
            nn.Softmax(dim=-1),
        )
        self.XANES =nn.ModuleList([MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,100,mlp_hidden_dims,mode='NAD',
                                            activation=nn.ReLU,normalization=nn.BatchNorm1d,dropout_rate=mlp_dropout,bias=True),
                                   nn.Linear(normnn_dim,100)])
        self.EXAFS = nn.ModuleList([MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,500,mlp_hidden_dims,mode='NAD',
                                             activation=nn.ReLU,normalization=nn.BatchNorm1d,dropout_rate=mlp_dropout,bias=True),
                                   nn.Linear(normnn_dim,500)])
        # self.xas_generator = nn.Sequential(
        #     ResidualUnit(1,1,16,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     ResidualUnit(1,16,32,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     ResidualUnit(1,32,2,adn_ordering='NDA',activation=nn.ReLU,normalization=nn.BatchNorm1d),
        #     )
        self.prompt_nn = nn.Linear(prompt_dim, prompt_hidden_dim)
        
        GNN_Transform = nn.ModuleList([
            nn.Linear(self.gnn_hidden_dims[i], self.gnn_hidden_dims[i + 1])
            for i in range(len(self.gnn_hidden_dims) - 1)
        ])
        self.GIN = nn.ModuleList([
            GINConv(apply_func=GNN_Transform[i],aggregator_type='sum',init_eps=0,learn_eps=False,activation=F.relu)
            for i in range(len(self.gnn_hidden_dims) - 1)
        ])
    def unify_vector(self,data:torch.Tensor):
        '''Unify the vector to [-1,1] range.'''
        max_vals=torch.max(data,dim=1,keepdim=True)[0]
        min_vals=torch.min(data,dim=1,keepdim=True)[0]
        return 2.0*(data-min_vals)/(max_vals-min_vals)-1.0
    def _generate_xas(self,atom_features,prompt):
        norm_matrix=self.NormalizeNN(atom_features)
        if self.xas_type == 'XANES':
            spectrum = self.XANES[0](torch.cat([atom_features, prompt], dim=-1))
            norm_matrix=self.XANES[1](norm_matrix)
        elif self.xas_type == 'EXAFS':
            spectrum = self.EXAFS[0](torch.cat([atom_features, prompt], dim=-1))
            norm_matrix=self.EXAFS[1](atom_features)
        spectrum=self.unify_vector(spectrum)
        spectrum=spectrum*norm_matrix
        return spectrum
    def forward(self, data, device):
        graph = data['graph'].to(device)
        prompt = data['prompt'].to(device)
        prompt = self.prompt_nn(prompt)
        atom_features = graph.ndata['feat']
        with graph.local_scope():
            for i in range(len(self.GIN)):
                result = self.GIN[i](graph, atom_features)
                if self.gnn_res_connection and i != 0:
                    atom_features = result + atom_features
                else:
                    atom_features = result
            graph.ndata['feat'] = atom_features
            atom_features = dgl.sum_nodes(graph, 'feat')
        spectrum=self._generate_xas(atom_features,prompt)
        return spectrum

class CrystalXASV3(CrystalXASV2):
    def __init__(self,
                 gnn_hidden_dims: Union[list,int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int=1024,
                 normnn_hidden_dim: list=[5120,2048],
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__(gnn_hidden_dims,gnn_layers,gnn_res_connection,feat_dim,prompt_dim,prompt_hidden_dim,normnn_dim,mlp_hidden_dims,mlp_dropout,xas_type)
        self.NormalizeNN=nn.Sequential(
            MLPBlock(dim_input=self.gnn_hidden_dims[-1],
                     dim_output=normnn_dim,
                     hidden_sizes=normnn_hidden_dim,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True),
            nn.Softmax(dim=-1),
        )

class CrystalXASV4(CrystalXASV2):
    def __init__(self,
                 gnn_hidden_dims: Union[list,int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int=1024,
                 normnn_hidden_dim: list=[5120,2048],
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__(gnn_hidden_dims, gnn_layers, gnn_res_connection, feat_dim, prompt_dim, prompt_hidden_dim, normnn_dim, mlp_hidden_dims, mlp_dropout, xas_type)
        del self.NormalizeNN
        self.XANES =nn.ModuleList([MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,100,mlp_hidden_dims,mode='NAD',
                                            activation=nn.ReLU,normalization=nn.BatchNorm1d,dropout_rate=mlp_dropout,bias=True),
                                   nn.Linear(normnn_dim,100)])
        self.EXAFS = nn.ModuleList([MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,500,mlp_hidden_dims,mode='NAD',
                                             activation=nn.ReLU,normalization=nn.BatchNorm1d,dropout_rate=mlp_dropout,bias=True),
                                   nn.Linear(normnn_dim,500)])