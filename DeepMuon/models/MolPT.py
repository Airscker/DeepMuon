'''
Author: airscker
Date: 2023-10-03 18:10:01
LastEditors: airscker
LastEditTime: 2023-10-17 15:13:13
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch
import torch.nn.functional as F
from typing import Union
from torch import nn
from dgl.nn.pytorch import GINEConv
from .base import MLPBlock


class AtomEmbedding(nn.Module):

    def __init__(self,
                 atom_feat_dim=150,
                 bond_feat_dim=12,
                 emb_dim: int = 300,
                 gnn_layers: int = 5,
                 res_connection: Union[int,bool] = True,
                 mlp_dims=[]):
        super().__init__()
        # self.atom_embedding = nn.Embedding(118, emb_dim)
        # self.edge_embedding = nn.Embedding(4, emb_dim)
        self.res_connection = int(res_connection)
        self.atom_embedding = nn.Linear(atom_feat_dim, emb_dim)
        self.bond_embedding = nn.Linear(bond_feat_dim, emb_dim)
        self.reset_parameters()
        self.GNN = nn.ModuleList(
            [GINEConv(nn.Linear(emb_dim, emb_dim)) for _ in range(gnn_layers)])
        self.mlp = nn.Sequential(
            MLPBlock(emb_dim,
                     118,
                     mlp_dims,
                     mode='NDA',
                     activation=nn.ReLU,
                     normalization=nn.BatchNorm1d),
            # nn.Softmax(dim=-1),
            # nn.Sigmoid()
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.atom_embedding.weight.data)

    def forward(self, graph:dgl.DGLGraph, device):
        graph=graph.to(device)
        atom_feat = graph.ndata['atom_feat']
        edge_feat = graph.edata['bond_feat']
        atom_emb = self.atom_embedding(atom_feat)
        edge_emb = self.bond_embedding(edge_feat)
        if self.res_connection:
            res_feat=atom_emb
        with graph.local_scope():
            for i in range(len(self.GNN)):
                atom_emb = F.relu(self.GNN[i](graph, atom_emb, edge_emb))
                if self.res_connection and (i+1)%self.res_connection == 0:
                    atom_emb = atom_emb + res_feat
                    res_feat = atom_emb
            if self.res_connection:
                del res_feat
            # graph.ndata['atom_feat'] = atom_emb
            # graph_feat = dgl.sum_nodes(graph, 'atom_feat')
            atom_idx = self.mlp(atom_emb)
            return atom_idx

class AtomEmbeddingV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,graph:dgl.DGLGraph,device):
        pass
