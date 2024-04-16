'''
Author: airscker
Date: 2023-09-10 17:32:44
LastEditors: airscker
LastEditTime: 2024-04-04 00:08:02
Description: NULL

Copyright (C) 2023 by Deep Graph Library, All Rights Reserved. 
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import dgl
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union
from .base import MLPBlock
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import GINConv, GINEConv, GraphConv


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
                 gnn_hidden_dims: Union[list, int] = [128, 512],
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
                print(
                    'Residual connection is unavailable when `gnn_hidden_dims` is a list. That is, dimensions of GNN layers must be the same.'
                )
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
            spectrum = torch.cat([
                self.XANES(torch.cat([atom_features, prompt], dim=-1)),
                self.EXAFS(torch.cat([atom_features, prompt], dim=-1))
            ],
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
                 gnn_hidden_dims: Union[list, int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int = 1024,
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__()
        xas_types = {'XANES': 100, 'EXAFS': 500}
        assert xas_type in xas_types.keys(
        ), f"'xas_type' must be in {xas_types.keys()}, but {xas_type} was given."
        self.xas_type = xas_type
        if isinstance(gnn_hidden_dims, int):
            self.gnn_hidden_dims = [gnn_hidden_dims] * gnn_layers
            self.gnn_res_connection = gnn_res_connection
        else:
            if gnn_res_connection:
                print(
                    'Residual connection is unavailable when `gnn_hidden_dims` is a list. That is, dimensions of GNN layers must be the same.'
                )
            self.gnn_res_connection = False
            self.gnn_hidden_dims = gnn_hidden_dims
        self.gnn_hidden_dims = [feat_dim] + self.gnn_hidden_dims
        self.NormalizeNN = nn.Sequential(
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
        self.XANES = nn.ModuleList([
            MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,
                     100,
                     mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True),
            nn.Linear(normnn_dim, 100)
        ])
        self.EXAFS = nn.ModuleList([
            MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,
                     500,
                     mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True),
            nn.Linear(normnn_dim, 500)
        ])
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
            GINConv(apply_func=GNN_Transform[i],
                    aggregator_type='sum',
                    init_eps=0,
                    learn_eps=False,
                    activation=F.relu)
            for i in range(len(self.gnn_hidden_dims) - 1)
        ])

    def unify_vector(self, data: torch.Tensor):
        '''Unify the vector to [-1,1] range.'''
        max_vals = torch.max(data, dim=1, keepdim=True)[0]
        min_vals = torch.min(data, dim=1, keepdim=True)[0]
        return 2.0 * (data - min_vals) / (max_vals - min_vals) - 1.0

    def _generate_xas(self, atom_features, prompt):
        norm_matrix = self.NormalizeNN(atom_features)
        if self.xas_type == 'XANES':
            spectrum = self.XANES[0](torch.cat([atom_features, prompt],
                                               dim=-1))
            norm_matrix = self.XANES[1](norm_matrix)
        elif self.xas_type == 'EXAFS':
            spectrum = self.EXAFS[0](torch.cat([atom_features, prompt],
                                               dim=-1))
            norm_matrix = self.EXAFS[1](atom_features)
        spectrum = self.unify_vector(spectrum)
        spectrum = spectrum * norm_matrix
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
        spectrum = self._generate_xas(atom_features, prompt)
        return spectrum


class CrystalXASV3(CrystalXASV2):

    def __init__(self,
                 gnn_hidden_dims: Union[list, int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int = 1024,
                 normnn_hidden_dim: list = [5120, 2048],
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__(gnn_hidden_dims, gnn_layers, gnn_res_connection,
                         feat_dim, prompt_dim, prompt_hidden_dim, normnn_dim,
                         mlp_hidden_dims, mlp_dropout, xas_type)
        self.NormalizeNN = nn.Sequential(
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
                 gnn_hidden_dims: Union[list, int] = [128, 512],
                 gnn_layers: int = 3,
                 gnn_res_connection: bool = True,
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 prompt_hidden_dim=32,
                 normnn_dim: int = 1024,
                 normnn_hidden_dim: list = [5120, 2048],
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__(gnn_hidden_dims, gnn_layers, gnn_res_connection,
                         feat_dim, prompt_dim, prompt_hidden_dim, normnn_dim,
                         mlp_hidden_dims, mlp_dropout, xas_type)
        del self.NormalizeNN
        self.XANES = nn.Sequential(
            MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,
                     100,
                     mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True), nn.Softmax(dim=-1), nn.Linear(100, 100))
        self.EXAFS = nn.Sequential(
            MLPBlock(self.gnn_hidden_dims[-1] + prompt_hidden_dim,
                     500,
                     mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True), nn.Softmax(dim=-1), nn.Linear(500, 500))

    def _generate_xas(self, atom_features, prompt):
        if self.xas_type == 'XANES':
            spectrum = self.XANES(torch.cat([atom_features, prompt], dim=-1))
        elif self.xas_type == 'EXAFS':
            spectrum = self.EXAFS(torch.cat([atom_features, prompt], dim=-1))
        return spectrum


class XASStructure(nn.Module):
    '''
    ## Predict XAS spectrums based on given crystal periodic structures

    ### Tips:
        
    '''

    def __init__(self,
                 xas_type='XANES',
                 atomic_num_dim: int = 118,
                 atom_coord_dim: int = 3,
                 bond_length_dim: int = 1,
                 gnn_layers: int = 3,
                 gnn_hidden_dims: Union[int, list] = [256, 256],
                 mlp_hidden_dims: Union[int, list] = [],
                 mlp_dropout: float = 0,
                 eps: float = 1e-3,
                 learn_eps: bool = True,
                 learnable_exp=True,
                 learnable_eps=True):
        super().__init__()
        self.xas_types = {'XANES': 100, 'EXAFS': 500, 'XAFS': 600}
        assert xas_type in self.xas_types.keys(
        ), f"'xas_type' must be in {self.xas_types.keys()}, but {xas_type} was given."
        self.xas_type = xas_type
        if isinstance(gnn_hidden_dims, int):
            gnn_hidden_dims = [gnn_hidden_dims] * gnn_layers
        self.gnn_hidden_dims = gnn_hidden_dims

        self.atom_embedding = nn.Linear(atomic_num_dim,
                                        self.gnn_hidden_dims[0])
        self.coord_embedding = nn.Linear(atom_coord_dim,
                                         self.gnn_hidden_dims[0])
        self.node_embedding = nn.Linear(2 * self.gnn_hidden_dims[0],
                                        self.gnn_hidden_dims[0])
        self.edge_embedding = nn.Linear(bond_length_dim,
                                        self.gnn_hidden_dims[0])

        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = self.register_buffer('eps', torch.Tensor([eps]))
        self.mlp = nn.Sequential(
            MLPBlock(dim_input=self.gnn_hidden_dims[-1],
                     dim_output=self.xas_types[xas_type],
                     hidden_sizes=mlp_hidden_dims,
                     mode='NAD',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d,
                     dropout_rate=mlp_dropout,
                     bias=True),
            nn.Sigmoid(),
        )
        # self.mlp = MLPBlock(self.gnn_hidden_dims[-1],
        #                     self.xas_types[xas_type],
        #                     mlp_hidden_dims,
        #                     mode='NAD',
        #                     activation=nn.ReLU,
        #                     normalization=nn.BatchNorm1d,
        #                     dropout_rate=mlp_dropout,
        #                     bias=True)
        # self.gnn = nn.ModuleList([
        #     GINEConv(apply_func=nn.Linear(self.gnn_hidden_dims[i],
        #                                  self.gnn_hidden_dims[i + 1]),
        #             init_eps=0,
        #             learn_eps=True)
        #     for i in range(len(self.gnn_hidden_dims) - 1)
        # ])
        self.gnn = nn.ModuleList([
            RRGraphConv(self.gnn_hidden_dims[i], self.gnn_hidden_dims[i + 1],learnable_eps=learnable_eps,learnable_exp=learnable_exp)
            for i in range(len(self.gnn_hidden_dims) - 1)
        ])

    def unify_vector(self, data: torch.Tensor):
        '''Unify the vector to [-1,1] range.'''
        max_vals = torch.max(data, dim=1, keepdim=True)[0]
        min_vals = torch.min(data, dim=1, keepdim=True)[0]
        return (data - min_vals) / (self.eps + max_vals - min_vals)

    # def _generate_xas(self, atom_features):
    #     norm_matrix = self.NormalizeNN(atom_features)
    #     spectrum = self.mlp(atom_features)
    #     spectrum = self.unify_vector(spectrum)
    #     spectrum = spectrum * norm_matrix
    #     return spectrum

    def forward(self, input, device):
        '''
        input:[graph,spectrum,struct_prompt,absorbing_atom]
        '''
        graph: dgl.DGLGraph = input[0].to(device)
        # torch.save(graph, './graph.pth')
        with graph.local_scope():
            atomic_num_embedding = self.atom_embedding(
                graph.ndata['atomic_num'])
            coord_embedding = self.coord_embedding(graph.ndata['coord'])
            node_embedding = self.node_embedding(
                torch.cat([atomic_num_embedding, coord_embedding], dim=-1))
            # torch.save(node_embedding, './node_embed.pth')
            # bond_length_embedding = self.edge_embedding(
            #     graph.edata['length'].unsqueeze(-1))
            for i in range(len(self.gnn)):
                '''
                You cannot directly apply the RELU activation function to the output of the GNN layer,
                otherwise the output will be almmost all zeros (because positive values are rare for GIN).
                '''
                node_embedding = self.gnn[i](
                    graph,
                    node_embedding,
                    graph.edata['length'],
                )
            # print(node_embedding)
            # torch.save(node_embedding, './gnn_nodes.pth')
            node_embedding[graph.ndata['abs_mask'] == 0] = 0
            graph.ndata['_h'] = node_embedding
            feature = dgl.sum_nodes(graph, '_h')/graph.num_nodes() 
            # torch.save(feature,'./pre_spectrum.pth')
            # feature = self._generate_xas(feature)
            # feature = self.mlp(self.unify_vector(feature))
            feature=self.mlp(feature)
            return feature

class RRGraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 agg_func=None,
                 apply_func=None,
                 learnable_exp=False,
                 learnable_eps=False):
        super().__init__()
        self.agg_func=agg_func
        if in_dim!=out_dim and apply_func is None:
            self.apply_func=nn.Linear(in_dim,out_dim)
        else:
            self.apply_func=apply_func
        if learnable_exp:
            self.exp=nn.Parameter(torch.Tensor([-2]))
        else:
            self.register_buffer('exp',torch.Tensor([-2]))
        if learnable_eps:
            self.eps=nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps',torch.Tensor([0]))

    def _aggregate(self, edges):
        src_node_feat = edges.src['_h']
        edge_weight = edges.data['_r']
        _msg = torch.ones_like(torch.exp(torch.log(edge_weight) * self.exp).view(-1, 1)) * src_node_feat
        if self.agg_func is not None:
            _msg = self.agg_func(_msg)
        return {'m': _msg}

    def _reducer(self,nodes):
        return {'_h':nodes.mailbox['m'].sum(dim=1)}
    def forward(self,graph:dgl.DGLGraph,feat:torch.Tensor,radius:torch.Tensor):
        with graph.local_scope():
            feat_src,feat_dst=expand_as_pair(feat,graph)
            graph.srcdata['_h']=feat_src
            graph.edata['_r']=radius
            graph.update_all(self._aggregate,self._reducer)
            rst=(1+self.eps)*feat_dst+graph.dstdata['_h']
            if self.apply_func is not None:
                rst=self.apply_func(rst)
            return rst

class XAS_Atom(nn.Module):

    def __init__(self,
                 xas_type: str = 'XANES',
                 hidden_dims: Union[int, list] = [],
                 dropout: float = 0) -> None:
        super().__init__()
        self.xas_length = {'XANES': 100, 'EXAFS': 500, 'XAFS': 600}
        self.mlp = MLPBlock(dim_input=self.xas_length[xas_type],
                            dim_output=118,
                            hidden_sizes=hidden_dims,
                            mode='NAD',
                            activation=nn.ReLU,
                            normalization=nn.BatchNorm1d,
                            dropout_rate=dropout,
                            bias=True)
        # self.norm=nn.Softmax(dim=-1)
    def forward(self, input):
        '''
        input.shape: N,L
        '''
        return self.mlp(input)


class XAS_Mask_Structure(nn.Module):

    def __init__(self,
                 atomic_num_dim: int = 118,
                 atom_coord_dim: int = 3,
                 bond_length_dim: int = 1,
                 num_layers: int = 3,
                 gnn_embedding_dims: Union[int, list] = [256, 256],
                 mlp_hidden_dims: Union[int, list] = [],
                 mlp_dropout: float = 0,
                 eps: float = 1e-3,
                 learn_eps: bool = True,
                 mask_length: int = 10) -> None:
        super().__init__()
        if isinstance(gnn_embedding_dims, int):
            gnn_embedding_dims = [gnn_embedding_dims] * num_layers
        assert len(
            gnn_embedding_dims
        ) == num_layers, f"The length of `gnn_embedding_dims` must be equal to `num_layers`, but {len(gnn_embedding_dims)} was given."
        self.node_embedding = nn.Linear(atomic_num_dim, gnn_embedding_dims[0])
        self.edge_embedding = nn.Linear(bond_length_dim, gnn_embedding_dims[0])
        self.gnn = nn.ModuleList([
            GINEConv(apply_func=nn.Linear(gnn_embedding_dims[i],
                                          gnn_embedding_dims[i + 1]),
                     learn_eps=True)
            for i in range(len(gnn_embedding_dims) - 1)
        ])
        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = self.register_buffer('eps', torch.Tensor([eps]))
        self.xanes = MLPBlock(dim_input=gnn_embedding_dims[-1],
                              dim_output=100,
                              hidden_sizes=mlp_hidden_dims,
                              bias=True,
                              dropout_rate=mlp_dropout)

    def unify_vector(self, data: torch.Tensor):
        '''Unify the vector to [0,1] range.'''
        max_val = torch.max(data)
        min_val = torch.min(data)
        return (data - min_val) / (self.eps + max_val - min_val)

    def forward(self, input, device):
        '''
        input:[graph,struct_prompt,masked_spectrum,absorbing_atom]
        '''
        graph: dgl.DGLGraph = input[0].to(device)
        with graph.local_scope():
            atomic_num_embedding = self.node_embedding(
                graph.ndata['atomic_num'])
            bond_length_embedding = self.edge_embedding(
                graph.edata['length'].unsqueeze(-1))
            for i in range(len(self.gnn)):
                atomic_num_embedding = self.gnn[i](graph, atomic_num_embedding,
                                                   bond_length_embedding)
            graph.ndata['_h'] = atomic_num_embedding
            feature = dgl.sum_nodes(graph, '_h')
            feature = self.xanes(self.unify_vector(feature))
            return feature
