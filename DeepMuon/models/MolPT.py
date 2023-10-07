'''
Author: airscker
Date: 2023-10-03 18:10:01
LastEditors: airscker
LastEditTime: 2023-10-04 15:11:20
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch import GINEConv
from .base import MLPBlock


class AtomEmbedding(nn.Module):

    def __init__(self,
                 emb_dim: int = 300,
                 gnn_layers: int = 5,
                 mlp_dims=[]):
        super().__init__()
        self.atom_embedding = nn.Embedding(118, emb_dim)
        self.edge_embedding = nn.Embedding(4, emb_dim)
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
            nn.Softmax(dim=-1),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.atom_embedding.weight.data)

    def forward(self, graph:dgl.DGLGraph, device):
        graph=graph.to(device)
        atom_feat = graph.ndata['atom_feat'].type(torch.int64)
        edge_feat = graph.edata['bond_feat'].type(torch.int64)
        atom_emb = self.atom_embedding(atom_feat)
        edge_emb = self.edge_embedding(edge_feat)
        with graph.local_scope():
            for i in range(len(self.GNN)):
                atom_emb = F.relu(self.GNN[i](graph, atom_emb, edge_emb))
            graph.ndata['atom_feat'] = atom_emb
            graph_feat = dgl.sum_nodes(graph, 'atom_feat')
            atom_idx = self.mlp(graph_feat)
            return atom_idx
