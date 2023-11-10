'''
Author: airscker
Date: 2023-10-24 18:57:33
LastEditors: airscker
LastEditTime: 2023-11-02 23:15:37
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import math
import os
import dgl
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Union
from functools import partial
from dgl.nn.pytorch import GINEConv,GraphConv,GINConv
from .base import MLPBlock


class MolSpaceMultiHeadAttention(nn.Module):

    def __init__(self,
                 embedding_dim: int = 1024,
                 num_heads=8,
                 dim_k: int = None,
                 dim_v: int = None,
                 attn_type: str = 'single',
                 aggragate='mean') -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        if dim_k is None:
            dim_k = embedding_dim
        if dim_v is None:
            dim_v = embedding_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        if aggragate not in ['sum', 'mean', 'max']:
            raise ValueError('aggragate must be one of sum,mean,max')
        if aggragate == 'sum':
            self.aggragate = partial(torch.sum, dim=1)
        elif aggragate == 'mean':
            self.aggragate = partial(torch.mean, dim=1)
        elif aggragate == 'max':
            self.aggragate = partial(torch.max, dim=1)

        # self.linear_q = nn.Linear(embedding_dim, dim_k * num_heads, bias=False)
        # self.linear_k = nn.Linear(embedding_dim, dim_k * num_heads, bias=False)
        # self.linear_v = nn.Linear(embedding_dim, dim_v * num_heads, bias=False)
        # self._norm_factor = 1 / math.sqrt(dim_k)
        self.linear_q = nn.Linear(embedding_dim, dim_k, bias=False)
        self.linear_k = nn.Linear(embedding_dim, dim_k, bias=False)
        self.linear_v = nn.Linear(embedding_dim, dim_v, bias=False)
        self._norm_factor = 1 / math.sqrt(dim_k // num_heads)
        self.attn_type = attn_type

    def _single_attention(self, data: torch.Tensor):
        '''
        data.shape: [num_tokens, embedding_dim]
        '''
        num_tokens, embedding_dim = data.shape
        dk = self.dim_k // self.num_heads
        dv = self.dim_v // self.num_heads
        q = self.linear_q(data).reshape(num_tokens, self.num_heads, dk)
        k = self.linear_k(data).reshape(num_tokens, self.num_heads, dk)
        v = self.linear_v(data).reshape(num_tokens, self.num_heads, dv)
        feat1 = torch.matmul(q, k.transpose(1, 2)) * self._norm_factor
        feat2 = torch.matmul(torch.softmax(feat1, dim=-1), v)
        attention = feat2.reshape(num_tokens, self.dim_v)
        # attention = self.aggragate(feat2)
        return attention

    def _batch_attention(self, x):
        '''
        x.shape: [batch_size, num_tokens, embedding_dim]
        '''
        batch_size, num_tokens, embedding_dim = x.shape
        dk = self.dim_k // self.num_heads
        dv = self.dim_v // self.num_heads
        q = self.linear_q(x).reshape(batch_size, num_tokens, self.num_heads,
                                     dk)
        k = self.linear_k(x).reshape(batch_size, num_tokens, self.num_heads,
                                     dk)
        v = self.linear_v(x).reshape(batch_size, num_tokens, self.num_heads,
                                     dv)
        feat1 = torch.matmul(q, k.transpose(2, 3)) * self._norm_factor
        feat2 = torch.matmul(torch.softmax(feat1, dim=-1), v)
        attention = feat2.reshape(batch_size, num_tokens, self.dim_v)
        # attention = self.aggragate(feat2)
        return attention

    def forward(self, x: list[torch.Tensor]):
        '''
        x.shape: [batch_size, num_tokens, embedding_dim]
        '''
        batch_size = len(x)
        attentions = []
        for i in range(batch_size):
            attentions.append(self._single_attention(x[i]))
        return attentions

class MolSpaceGNNFeaturizer(nn.Module):
    def __init__(self,
                 pretrained_path:str='',
                 freeze_gnn=False,
                 atom_feat_dim=150,
                 bond_feat_dim=12,
                 emb_dim: int = 1024,
                 gnn_layers: int = 20,
                 res_connection: Union[int,bool] = 2,
                 ) -> None:
        super().__init__()
        self.freeze_gnn = freeze_gnn
        self.res_connection = int(res_connection)
        self.atom_embedding = nn.Linear(atom_feat_dim, emb_dim)
        self.bond_embedding = nn.Linear(bond_feat_dim, emb_dim)
        self.GNN = nn.ModuleList([
            GINConv(nn.Linear(emb_dim, emb_dim)) for _ in range(gnn_layers)
        ])
        # self.GNN = nn.ModuleList([
        #     GraphConv(emb_dim, emb_dim, norm='both', activation=F.relu,allow_zero_in_degree=True) for _ in range(gnn_layers)
        # ])
        if os.path.exists(pretrained_path):
            self.pre_trained_path = pretrained_path
            self.load_pretrained()
    def load_pretrained(self):
        unloaded=[]
        state_dict=torch.load(self.pre_trained_path,map_location='cpu')['model']
        for name, para in self.named_parameters():
            if name in state_dict:
                try:
                    para.data = state_dict[name]
                except:
                    unloaded.append(name)
        if len(unloaded)>0:
            print(f'Unloaded pretrained model parameters: {unloaded}')
        else:
            print('Load pretrained model successfully!')
    def _freeze_gnn(self):
        self.GNN.eval()
        for param in self.GNN.parameters():
            param.requires_grad = False

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
                # atom_emb = F.relu(self.GNN[i](graph, atom_emb, edge_emb))
                # atom_emb=self.GNN[i](graph,atom_emb)
                atom_emb = F.relu(self.GNN[i](graph, atom_emb))
                if self.res_connection and (i+1)%self.res_connection == 0:
                    atom_emb = atom_emb + res_feat
                    res_feat = atom_emb
            if self.res_connection:
                del res_feat
            return atom_emb
    def train(self,mode: bool = True):
        super().train(mode)
        if self.freeze_gnn:
            self._freeze_gnn()

class MolSpaceTransformer(nn.Module):
    def __init__(self, embedding_dim:int=1024,num_heads:int=8,) -> None:
        super().__init__()
        self.multi_head_attention=MolSpaceMultiHeadAttention(embedding_dim=embedding_dim,num_heads=num_heads)
        self.mlp=nn.Linear(embedding_dim,embedding_dim)
        self.norm1=nn.LayerNorm(embedding_dim)
        self.norm2=nn.LayerNorm(embedding_dim)
    def forward(self, x:list[torch.Tensor]):
        '''
        x.shape: [batch_size, num_tokens, embedding_dim]
        '''
        batch_size=len(x)
        attentions=self.multi_head_attention(x)
        for i in range(batch_size):
            attn=self.norm1(attentions[i]+x[i])
            attn=self.norm2(self.mlp(attn)+attn)
            x[i]=attn
        return x

class MolSpace(nn.Module):
    def __init__(self,
                 classes:int=1,
                 add_dim:int=0,
                 mlp_dims:list[int]=[],
                 dropout:float=0.0,
                 embedding_dim:int=1024,
                 num_heads=8,
                 attn_depth:int=10,
                 aggragate:str='sum',
                 pretrained_path:str='',
                 freeze_gnn=False,
                 atom_feat_dim=150,
                 bond_feat_dim=12,
                 gnn_layers: int = 20,
                 gnn_res_connection: Union[int,bool] = 2,
                 ) -> None:
        super().__init__()
        self.add_dim=add_dim
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.attn_depth=attn_depth
        if aggragate not in ['sum','mean','max']:
            raise ValueError('aggragate must be one of sum,mean,max')
        if aggragate=='sum':
            self.aggragate=partial(torch.sum,dim=0)
        elif aggragate=='mean':
            self.aggragate=partial(torch.mean,dim=0)
        elif aggragate=='max':
            self.aggragate=partial(torch.max,dim=0)
        self.gnn_featurizer=MolSpaceGNNFeaturizer(pretrained_path=pretrained_path,
                                                  freeze_gnn=freeze_gnn,
                                                  atom_feat_dim=atom_feat_dim,
                                                  bond_feat_dim=bond_feat_dim,
                                                  emb_dim=embedding_dim,
                                                  gnn_layers=gnn_layers,
                                                  res_connection=gnn_res_connection)
        self.transformers=nn.ModuleList([
            MolSpaceTransformer(embedding_dim=embedding_dim,num_heads=num_heads) for _ in range(attn_depth)
        ])
        self.mlp=MLPBlock(dim_input=embedding_dim+add_dim,
                          dim_output=classes,
                          hidden_sizes=mlp_dims,
                          activation=nn.ReLU,
                          dropout_rate=dropout)
    def forward(self,data:list,device:Union[str,torch.device]='cpu'):
        '''
        graphs: list of dgl.DGLGraph
        '''
        graphs=data['graphs']
        batch_size=len(graphs)
        node_feat=[]
        aggragated_feat=[]
        for i in range(batch_size):
            node_feat.append(self.gnn_featurizer(graphs[i],device))
        for i in range(self.attn_depth):
            node_feat=self.transformers[i](node_feat)
        for i in range(batch_size):
            aggragated_feat.append(self.aggragate(node_feat[i]))
        aggragated_feat=torch.stack(aggragated_feat)
        if self.add_dim>0:
            add_feat=data['add_features']
            add_feat=add_feat.to(device)
            print(aggragated_feat.shape,add_feat.shape)
            aggragated_feat=torch.cat([aggragated_feat,add_feat],dim=-1)
        output=self.mlp(aggragated_feat)
        return output.squeeze(-1)
