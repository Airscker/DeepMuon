'''
Author: airscker
Date: 2023-05-23 14:36:30
LastEditors: airscker
LastEditTime: 2023-12-14 01:38:46
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, NNConv, GINConv

import numpy as np
from rdkit import Chem
from typing import Union
from torch_geometric.data import Data
from .base import MLPBlock,GNN_feature

class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.LeakyReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.LeakyReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, graph, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(graph, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats

class SolvGNN(nn.Module):
    def __init__(self, in_dim=74, hidden_dim=256, add_dim=33,edge_hidden_dim=512,n_classes=1):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim,allow_zero_in_degree=True)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=edge_hidden_dim,
                                     num_step_message_passing=1)
        # self.global_conv1=GraphConv(hidden_dim+4,hidden_dim)

        # self.add_embed_dims=[128,256,64]
        # self.add_embed=nn.Sequential(
        #     nn.Linear(add_dim+2,self.add_embed_dims[0]),
        #     nn.BatchNorm1d(self.add_embed_dims[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.add_embed_dims[0],self.add_embed_dims[1]),
        #     nn.BatchNorm1d(self.add_embed_dims[1]),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.add_embed_dims[1],self.add_embed_dims[2])
        # )
        self.regression=MLPBlock(257,n_classes,[hidden_dim]*2,mode='NAD',activation=nn.LeakyReLU)

    # def forward(self, solvdata=None,empty_solvsys=None,device=None):
    #     graph:dgl.DGLGraph=solvdata['graph'].to(device)
    #     with graph.local_scope():
    #         graph_ndata=graph.ndata['h'].float().to(device)
    #         graph_edata=graph.edata['type'].float().to(device)
    #         inter_hb=solvdata['inter_hb'][:,None].float().to(device)
    #         # be_salt=solvdata['be_salt'][:,None].to(device)
    #         # be_ps=solvdata['be_ps'][:,None].to(device)
    #         # ip=solvdata['ip'][:,None].to(device)
    #         add_feature=solvdata['add_features'].float().to(device)
    #         graph.ndata['h']=F.relu(self.conv2(graph,(F.relu(self.conv1(graph,(graph_ndata,graph_edata))),graph_edata)))
    #         # graph_mean=torch.cat([graph_mean,inter_hb,be_salt,be_ps,ip],axis=1)
    #         # graph_mean=torch.cat([graph_mean,graph_mean])
    #         # inter_feature=torch.cat([inter_hb,be_salt,be_ps,ip],axis=0)
    #         # gh_feature=self.global_conv1(empty_solvsys,node_mean,edge_mean)
    #         graph.ndata['h']=self.global_conv1(graph,graph.ndata['h'],graph_edata)
    #         # print(gh_feature.shape,dgl.mean_nodes(graph,'h').shape,dgl.mean_edges(graph,'type').shape)
    #         node_mean=dgl.mean_nodes(graph,'h')
    #         edge_mean=dgl.mean_edges(graph,'type')
    #         # gh_feature=torch.cat([node_mean,edge_mean,inter_hb,be_salt,be_ps,ip],axis=1)
    #         add_feature=torch.cat([edge_mean,inter_hb,add_feature],axis=1)
    #         add_feature=self.add_embed(add_feature)
    #         gh_feature=torch.cat([node_mean,add_feature],axis=1)
    #         output=self.regression(gh_feature)
    #         # output = torch.cat((output[0:len(output)//2,:],output[len(output)//2:,:]),axis=1)
    #         # output=torch.mean(output,dim=1).unsqueeze(1)
    #         return output
    def forward(self, solvdata=None,empty_solvsys=None,device=None):
        graph:dgl.DGLGraph=solvdata['graph'].to(device)
        with graph.local_scope():
            graph_ndata=graph.ndata['h'].float().to(device)
            graph_edata=graph.edata['type'].float().to(device)
            inter_hb=solvdata['inter_hb'][:,None].float().to(device)
            graph.ndata['h']=F.relu(self.conv2(graph,(F.relu(self.conv1(graph,(graph_ndata,graph_edata))),graph_edata)))
            graph.ndata['h']=self.global_conv1(graph,graph.ndata['h'],graph_edata)
            node_mean=dgl.mean_nodes(graph,'h')
            edge_mean=dgl.mean_edges(graph,'type')
            # add_feature=torch.cat([edge_mean,inter_hb],axis=1)
            # add_feature=self.add_embed(add_feature)
            gh_feature=torch.cat([node_mean,edge_mean],axis=1)
            output=self.regression(gh_feature)
            return output.squeeze(-1)

class SolvGNNV2(nn.Module):
    def __init__(self, in_dim=74, hidden_dim=256, add_dim=33,edge_hidden_dim=512,n_classes=1):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim,allow_zero_in_degree=True)
        # self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
        #                              edge_in_feats=1,
        #                              node_out_feats=hidden_dim,
        #                              edge_hidden_feats=edge_hidden_dim,
        #                              num_step_message_passing=1)
        # self.global_conv1=GraphConv(hidden_dim,hidden_dim)

        # self.add_embed_dims=[128,256,64]
        # self.add_embed=nn.Sequential(
        #     nn.Linear(add_dim+2,self.add_embed_dims[0]),
        #     nn.BatchNorm1d(self.add_embed_dims[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.add_embed_dims[0],self.add_embed_dims[1]),
        #     nn.BatchNorm1d(self.add_embed_dims[1]),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.add_embed_dims[1],self.add_embed_dims[2])
        # )
        self.hidden_dims=[1024,512]
        self.regression=MLPBlock(hidden_dim,n_classes,self.hidden_dims,mode='NAD',activation=nn.LeakyReLU)

    def forward(self, solvdata=None,empty_solvsys=None,device=None):
        graph:dgl.DGLGraph=solvdata['graph'].to(device)
        with graph.local_scope():
            graph_ndata=graph.ndata['h'].float().to(device)
            # graph_edata=graph.edata['type'].float().to(device)
            # inter_hb=solvdata['inter_hb'][:,None].float().to(device)
            graph.ndata['h']=F.relu(self.conv2(graph,F.relu(self.conv1(graph,graph_ndata))))
            # graph.ndata['h']=self.global_conv1(graph,graph.ndata['h'],graph_edata)
            node_mean=dgl.mean_nodes(graph,'h')
            # node_mean=self.global_conv1(empty_solvsys,node_mean)
            # edge_mean=dgl.mean_edges(graph,'type')
            # add_feature=torch.cat([edge_mean,inter_hb],axis=1)
            # add_feature=self.add_embed(add_feature)
            # gh_feature=torch.cat([node_mean,edge_mean],axis=1)
            # print(gh_feature.shape)
            output=self.regression(node_mean)
            return output.squeeze(-1)

class GCR(nn.Module):
    def __init__(self,dim=256,allow_zero_in_degree=True) -> None:
        super().__init__()
        self.gcn1=GraphConv(dim, dim,allow_zero_in_degree=allow_zero_in_degree)
        self.gcn2=GraphConv(dim, dim,allow_zero_in_degree=allow_zero_in_degree)
    def forward(self,graph:dgl.DGLGraph,node_feature:torch.Tensor,bond_feature:torch.Tensor=None):
        with graph.local_scope():
            output=self.gcn1(graph,node_feature)
            output=self.gcn2(graph,F.relu(output))
            # return node_feature+F.relu(output)
            return F.relu(output)

class SolvGNNV3(nn.Module):
    def __init__(self, in_dim=74, hidden_dim=256, add_dim=0,
                 mlp_dims=[1024,512],dropout=0, gcr_layers=5,
                 n_classes=1, res_connection:Union[int,bool]=0,
                 allow_zero_in_degree=True,freeze_GNN=False) -> None:
        super().__init__()
        self.add_dim=add_dim
        self.res_connection=int(res_connection)
        self.freeze_GNN=freeze_GNN
        # self.node_embedding=nn.Linear(in_dim,hidden_dim)
        self.gcn=GraphConv(in_dim, hidden_dim,allow_zero_in_degree=allow_zero_in_degree)
        self.node_gcr=nn.ModuleList(
            [GCR(dim=hidden_dim,allow_zero_in_degree=allow_zero_in_degree) for _ in range(gcr_layers)]
        )
        self.regression=MLPBlock(hidden_dim+add_dim,n_classes,mlp_dims,mode='NAD',activation=nn.LeakyReLU,dropout_rate=dropout)
        # self.regression = nn.Sequential(
        #     nn.Linear(hidden_dim+add_dim, mlp_dims[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(mlp_dims[0],mlp_dims[1]),
        #     nn.LeakyReLU(),
        #     nn.Linear(mlp_dims[1],n_classes)
        # )
        # self.regression=nn.Linear(hidden_dim+add_dim,n_classes)
    def forward(self,solvdata=None,empty_solvsys=None,device=None):
        graph:dgl.DGLGraph=solvdata['graph'].to(device)
        if self.add_dim>0:
            add_features=solvdata['add_features'].float().to(device)
            add_features=add_features.squeeze()
        with graph.local_scope():
            graph_ndata=graph.ndata['h'].float().to(device)
            node_feature=self.gcn(graph,graph_ndata)
            # node_feature=self.node_embedding(graph_ndata)
            for i in range(len(self.node_gcr)):
                feature=self.node_gcr[i](graph,node_feature)
                if self.res_connection>0 and i>0 and (i+1)%self.res_connection==0:
                    node_feature=node_feature+feature
                else:
                    node_feature=feature
            graph.ndata['h']=node_feature
            node_mean=dgl.mean_nodes(graph,'h')
            # print(node_mean.shape,add_features.shape)
            if self.add_dim>0:
                node_mean=torch.cat([node_mean,add_features],axis=1)
            output=self.regression(node_mean).squeeze(-1)
            return output
    def freeze_GNNPart(self):
        if self.freeze_GNN:
            for para in self.node_gcr.parameters():
                para.requires_grad=False
            for para in self.gcn.parameters():
                para.requires_grad=False
    def train(self,mode=True):
        super().train(mode)
        self.freeze_GNNPart()


class SolvLinear(nn.Module):
    def __init__(self, in_dims:int) -> None:
        super().__init__()
        self.linear=nn.Linear(in_dims,1)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0)
    def forward(self,data):
        return self.linear(data).squeeze(-1)


class SolvGNNV4(nn.Module):
    def __init__(self, mlp_dropout=0.2,mlp_hidden_dim=[1024,512]) -> None:
        super().__init__()
        self.mlp=MLPBlock(300,1,mlp_hidden_dim,mode='NAD',normalization=nn.BatchNorm1d,activation=nn.ReLU,dropout_rate=mlp_dropout)
    def forward(self,data):
        output=self.mlp(data)
        return output.unsqueeze(-1)

class SolvGNNV5(nn.Module):
    def __init__(self,
                 in_dim=74,
                 hidden_dim=2048,
                 add_dim=0,
                 mlp_dims=[1024, 512],
                 dropout_rate=0,
                 norm=False,
                 gcr_layers=25,
                 n_classes=1,
                 res_connection: Union[int, bool] = 0,
                 allow_zero_in_degree=True,
                 freeze_GNN=False) -> None:
        super().__init__()
        self.res_connection=int(res_connection)
        gnn_hidden_dims=[hidden_dim]*gcr_layers
        self.node_update=nn.ModuleList([
            nn.Linear(gnn_hidden_dims[i],gnn_hidden_dims[i+1]) for i in range(len(gnn_hidden_dims)-1)
        ])
        self.pre_gnn=GINConv(nn.Linear(in_dim,hidden_dim),aggregator_type='sum',init_eps=0,learn_eps=False,activation=F.relu)
        self.gnn=nn.ModuleList([
            GINConv(self.node_update[i],aggregator_type='sum',init_eps=0,learn_eps=False,activation=F.relu)
            for i in range(len(gnn_hidden_dims)-1)
        ])
        self.add_dim=add_dim
        self.freeze_GNN=freeze_GNN
        self.regression=MLPBlock(hidden_dim+add_dim,
                                 n_classes,
                                 mlp_dims,
                                 mode='NAD',
                                 activation=nn.LeakyReLU,
                                 normalization=nn.BatchNorm1d if norm else None,
                                 dropout_rate=dropout_rate)
    def forward(self,solvdata=None,empty_solvsys=None,device=None):
        graph:dgl.DGLGraph=solvdata['graph'].to(device)
        if self.add_dim>0:
            add_features=solvdata['add_features'].float().to(device)
            add_features=add_features.squeeze()
        with graph.local_scope():
            graph_ndata=graph.ndata['h'].float().to(device)
            graph_ndata=self.pre_gnn(graph,graph_ndata)
            if self.res_connection:
                res_feat=graph_ndata
            for i in range(len(self.gnn)):
                feature=self.gnn[i](graph,graph_ndata)
                if self.res_connection>0 and i>0 and (i+1)%self.res_connection==0:
                    graph_ndata=feature+res_feat
                    res_feat=graph_ndata
                else:
                    graph_ndata=feature
            if self.res_connection:
                del res_feat
            graph.ndata['h']=graph_ndata
            node_mean=dgl.mean_nodes(graph,'h')
            if self.add_dim>0:
                node_mean=torch.cat([node_mean,add_features],axis=1)
            output=self.regression(node_mean)
            return output.squeeze(-1)

class SolvGNNV6(nn.Module):

    def __init__(self,
                 in_dim=74,
                 hidden_dim=256,
                 add_dim=0,
                 mlp_dims=[1024, 512],
                 dropout=0,
                 gcr_layers=5,
                 n_classes=1,
                 res_connection: Union[int, bool] = 0,
                 allow_zero_in_degree=True,
                 freeze_GNN=False) -> None:
        super().__init__()
        self.add_dim = add_dim
        self.res_connection = int(res_connection)
        self.freeze_GNN = freeze_GNN
        self.gcn = GraphConv(in_dim,
                             hidden_dim,
                             allow_zero_in_degree=allow_zero_in_degree)
        self.node_gcr = nn.ModuleList([
            GCR(dim=hidden_dim, allow_zero_in_degree=allow_zero_in_degree)
            for _ in range(gcr_layers)
        ])
        self.regression = MLPBlock(2*hidden_dim + add_dim,
                                   n_classes,
                                   mlp_dims,
                                   mode='NAD',
                                   activation=nn.LeakyReLU,
                                   dropout_rate=dropout)
        # self.sum_weight = nn.Parameter(torch.Tensor([0.5, 0.5]),requires_grad=True)

    def _feat_extract(self,
                      graph: dgl.DGLGraph,
                      node_feature: torch.Tensor,
                      device=None):
        node_feature = node_feature.float().to(device)
        with graph.local_scope():
            output = self.gcn(graph, node_feature)
            for i in range(len(self.node_gcr)):
                feature = self.node_gcr[i](graph, output)
                if self.res_connection > 0 and i > 0 and (
                        i + 1) % self.res_connection == 0:
                    output = output + feature
                else:
                    output = feature
            graph.ndata['h'] = output
            node_mean = dgl.sum_nodes(graph, 'h')
            return node_mean,output

    def forward(self, solvdata=None, empty_solvsys=None, device=None):
        cation: dgl.DGLGraph = solvdata['cation'].to(device)
        anion: dgl.DGLGraph = solvdata['anion'].to(device)
        if self.add_dim > 0:
            add_features = solvdata['add_features'].float().to(device)
            add_features = add_features.squeeze()
        cation_feat,_ = self._feat_extract(cation, cation.ndata['h'], device)
        anion_feat,_ = self._feat_extract(anion, anion.ndata['h'], device)
        sumed_feat = self.sum_weight[0] * cation_feat + self.sum_weight[1] * anion_feat
        output = self.regression(
            torch.cat([cation_feat, anion_feat, add_features], axis=1)
            # torch.cat([sumed_feat, add_features], axis=1)
        )
        return output.squeeze(-1)
class Attention(nn.Module):
    def __init__(self,in_dim:int=1024,q_dim:int=None,v_dim:int=None) -> None:
        super().__init__()
        if q_dim is None:
            q_dim=in_dim
        if v_dim is None:
            v_dim=in_dim
        self.q_linear=nn.Linear(in_dim,q_dim)
        self.k_linear=nn.Linear(in_dim,q_dim)
        self.v_linear=nn.Linear(in_dim,v_dim)

    def forward(self,x:torch.Tensor):
        '''
        x.shape: (Node_num, in_dim)
        '''
        q=self.q_linear(x)
        k=self.k_linear(x)
        v=self.v_linear(x)
        attention=torch.softmax(torch.matmul(q,k.T),dim=1)
        output=torch.matmul(attention,v)
        return output

class SolvGNNV7(nn.Module):
    def __init__(self,
                 in_dim=74,
                 hidden_dim=256,
                 add_dim=0,
                 mlp_dims=[1024, 512],
                 dropout=0,
                 gcr_layers=5,
                 n_classes=1,
                 res_connection: Union[int, bool] = 0,
                 allow_zero_in_degree=True,
                 freeze_GNN=False) -> None:
        super().__init__()
        self.add_dim = add_dim
        self.res_connection = int(res_connection)
        self.freeze_GNN = freeze_GNN
        self.gcn = GraphConv(in_dim,
                             hidden_dim,
                             allow_zero_in_degree=allow_zero_in_degree)
        self.node_gcr = nn.ModuleList([
            GCR(dim=hidden_dim, allow_zero_in_degree=allow_zero_in_degree)
            for _ in range(gcr_layers)
        ])
        # self.attention=Attention(hidden_dim)
        # self.attention=nn.ModuleList([
        #     Attention(hidden_dim) for _ in range(gcr_layers)
        # ])
        self.regression = MLPBlock(hidden_dim + add_dim,
                                   n_classes,
                                   mlp_dims,
                                   mode='NAD',
                                   activation=nn.LeakyReLU,
                                   dropout_rate=dropout)
        # self.sum_weight = nn.Parameter(torch.Tensor([0.5, 0.5]),requires_grad=True)
        self.q_linear=nn.Linear(hidden_dim,hidden_dim)
        self.k_linear=nn.Linear(hidden_dim,hidden_dim)
        self.v_linear=nn.Linear(hidden_dim,1)
    def _weighted_molcule(self,x:torch.Tensor):
        q=self.q_linear(x)
        k=self.k_linear(x)
        v=self.v_linear(x)
        attention=torch.softmax(torch.matmul(q,k.T),dim=1)
        output=torch.matmul(attention,v)
        return output
    def _feat_extract(self,
                      graph: dgl.DGLGraph,
                      node_feature: torch.Tensor,
                      device=None):
        node_feature = node_feature.float().to(device)
        with graph.local_scope():
            output = self.gcn(graph, node_feature)
            for i in range(len(self.node_gcr)):
                feature = self.node_gcr[i](graph, output)
                if self.res_connection > 0 and i > 0 and (
                        i + 1) % self.res_connection == 0:
                    output = output + feature
                else:
                    output = feature
                # output=self.attention[i](output)
            graph.ndata['h'] = output
            node_mean = dgl.sum_nodes(graph, 'h')
            return node_mean,output

    def forward(self, solvdata=None, empty_solvsys=None, device=None):
        graphs = solvdata['graphs']
        features=[]
        for graph in graphs:
            graph: dgl.DGLGraph = graph.to(device)
            _,system_feat=self._feat_extract(graph, graph.ndata['h'], device)
            feat_weight=self._weighted_molcule(system_feat).squeeze(-1)
            features.append(torch.matmul(system_feat.T,feat_weight))
        features=torch.stack(features)
        if self.add_dim > 0:
            add_features = solvdata['add_features'].float().to(device)
            add_features = add_features.squeeze()
        output = self.regression(
            torch.cat([features, add_features], axis=1)
        )
        return output.squeeze(-1)

class AdjGNN(nn.Module):

    def __init__(self,
                 in_dim=74,
                 edge_dim=12,
                 hidden_dim=256,
                 add_dim=0,
                 mlp_dims=[1024, 512],
                 dropout=0,
                 gcr_layers=5,
                 n_classes=1,
                 res_connection: Union[int, bool] = 0,
                 allow_zero_in_degree=True,
                 freeze_GNN=False,
                 bias=True,
                 degree_norm=True,
                 weighted_sum=True,
                 transform=True,
                 attention=False,
                 no_adj=False,
                 cross_adj=False,
                 include_edge=False) -> None:
        super().__init__()
        self.node_embedding=nn.Linear(in_dim,hidden_dim)
        self.bond_embedding=nn.Linear(edge_dim,hidden_dim)
        self.node_gcr=nn.ModuleList([
            node_adjV2(hidden_dim,bias,nn.ReLU(),degree_norm,weighted_sum,transform,attention,no_adj,cross_adj,include_edge) for _ in range(gcr_layers)
        ])
        self.res_connection=int(res_connection)
        self.add_dim=add_dim
        self.freeze_GNN=freeze_GNN
        self.include_edge=include_edge
        self.regression=MLPBlock(hidden_dim+add_dim,n_classes,mlp_dims,mode='NAD',activation=nn.LeakyReLU,dropout_rate=dropout)
    def forward(self, solvdata=None, empty_solvsys=None, device=None):
        graph: dgl.DGLGraph = solvdata['graph'].to(device)
        if self.add_dim > 0:
            add_features = solvdata['add_features'].float().to(device)
            add_features = add_features.squeeze()
        with graph.local_scope():
            graph_ndata = graph.ndata['h'].float().to(device)
            node_feature = self.node_embedding(graph_ndata)
            if self.include_edge:
                graph_edata = graph.edata['e'].float().to(device)
                bond_feature=self.bond_embedding(graph_edata)
            else:
                bond_feature=None
            for i in range(len(self.node_gcr)):
                feature = self.node_gcr[i](graph, node_feature, bond_feature)
                if self.res_connection > 0 and i > 0 and (
                        i + 1) % self.res_connection == 0:
                    node_feature = node_feature + feature
                else:
                    node_feature = feature
            graph.ndata['h'] = node_feature
            node_mean = dgl.mean_nodes(graph, 'h')
            # print(node_mean.shape,add_features.shape)
            if self.add_dim > 0:
                node_mean = torch.cat([node_mean, add_features], axis=1)
            output = self.regression(node_mean).squeeze(-1)
            return output

# class node_adj(nn.Module):
#     def __init__(self,
#                  node_dim:int,
#                  bias=False,
#                  activation=None,
#                  degree_norm=True,
#                  weighted_sum=False,
#                  transform=True,
#                  attention=False,
#                  no_adj=False,
#                  cross_adj=False,
#                  include_edge=False) -> None:
#         super().__init__()
#         if attention:
#             self.attention_embedding=node_attention(node_dim)
#         else:
#             self.attention_embedding=None
#         self.embedding=nn.Linear(node_dim,node_dim)
#         if include_edge:
#             self.embedding_edge=nn.Linear(node_dim,node_dim)
#         else:
#             self.register_parameter('embedding_edge',None)
#         if bias:
#             self.bias=nn.Parameter(torch.Tensor(node_dim))
#         else:
#             self.register_parameter('bias',None)
#         self.degree_norm=degree_norm
#         if activation is not None:
#             self.activation=activation
#         if weighted_sum:
#             self.sum_weight=nn.Parameter(torch.Tensor([0.5,0.5]))
#         else:
#             self.register_buffer('sum_weight',torch.Tensor([1,1]))
#         if transform:
#             self.transform=nn.Linear(node_dim,node_dim)
#         else:
#             self.register_parameter('transform',None)
#         self.reset_parameters()
#         self.no_adj=no_adj
#         self.cross_adj=cross_adj
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.embedding.weight.data)
#         if self.bias is not None:
#             self.bias.data.fill_(0)
#     def forward(self,graph:dgl.DGLGraph,node_feature:torch.Tensor,bond_feature:torch.Tensor=None):
#         adj_matrix=graph.adjacency_matrix().to_dense()
#         msg=self.embedding(node_feature)
#         if not self.no_adj:
#             msg = torch.matmul(adj_matrix, msg)
#             if self.degree_norm:
#                 msg = msg / torch.sum(adj_matrix, axis=1, keepdim=True)
#         if self.attention_embedding is not None:
#             att_msg,att=self.attention_embedding(node_feature)
#             if self.cross_adj:
#                 att=att*adj_matrix
#                 att_msg=torch.matmul(att,node_feature)
#             msg=msg+att_msg
#         if self.bias is not None:
#             msg += self.bias
#         if hasattr(self,'activation'):
#             msg = self.activation(msg)
#         msg=self.sum_weight[0]*msg+self.sum_weight[1]*node_feature
#         if self.transform is not None:
#             msg = self.transform(msg)
#         return msg

class node_adjV2(nn.Module):

    def __init__(self,
                 node_dim: int,
                 bias=False,
                 activation=None,
                 degree_norm=True,
                 weighted_sum=False,
                 transform=True,
                 attention=False,
                 no_adj=False,
                 cross_adj=False,
                 include_edge=False) -> None:
        super().__init__()
        if attention:
            self.attention_embedding = node_attention(node_dim)
        else:
            self.attention_embedding = None
        self.embedding = nn.Linear(node_dim, node_dim)
        self.include_edge = include_edge
        if include_edge:
            self.embedding_edge = nn.Linear(node_dim, node_dim)
        else:
            self.register_parameter('embedding_edge', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(node_dim))
        else:
            self.register_parameter('bias', None)
        self.degree_norm = degree_norm
        if activation is not None:
            self.activation = activation
        if weighted_sum:
            self.sum_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        else:
            self.register_buffer('sum_weight', torch.Tensor([1, 1]))
        if transform:
            self.transform = nn.Linear(node_dim, node_dim)
        else:
            self.register_parameter('transform', None)
        self.reset_parameters()
        self.no_adj = no_adj
        self.cross_adj = cross_adj

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
    def _message_func(self,edges):
        src_node_feat=edges.src['_hf']
        # dst_node_feat=edges.dst['_h']
        if self.include_edge:
            edge_feat=edges.data['_ef']
            return {'msg':src_node_feat+edge_feat}
        else:
            return {'msg':src_node_feat}
    def _reduce_func(self,nodes):
        return {'_h':torch.sum(nodes.mailbox['msg'],dim=1)}

    def forward(self,
                graph: dgl.DGLGraph,
                node_feature: torch.Tensor,
                bond_feature: torch.Tensor = None):
        with graph.local_scope():
            adj_matrix = graph.adjacency_matrix().to_dense()
            graph.ndata['_hf'] = self.embedding(node_feature)
            if self.embedding_edge is not None:
                graph.edata['_ef']=self.embedding_edge(bond_feature)
            graph.update_all(self._message_func,self._reduce_func)
            msg=graph.ndata['_h']
            if self.degree_norm:
                msg = msg / torch.sum(adj_matrix, axis=1, keepdim=True)
            if self.bias is not None:
                msg += self.bias
            if hasattr(self, 'activation'):
                msg = self.activation(msg)
            msg = self.sum_weight[0] * msg + self.sum_weight[1] * node_feature
            if self.transform is not None:
                msg = self.transform(msg)
            return msg
        # if not self.no_adj:
        #     msg = torch.matmul(adj_matrix, msg)
        #     if self.degree_norm:
        #         msg = msg / torch.sum(adj_matrix, axis=1, keepdim=True)
        # if self.attention_embedding is not None:
        #     att_msg, att = self.attention_embedding(node_feature)
        #     if self.cross_adj:
        #         att = att * adj_matrix
        #         att_msg = torch.matmul(att, node_feature)
        #     msg = msg + att_msg
        # if self.bias is not None:
        #     msg += self.bias
        # if hasattr(self, 'activation'):
        #     msg = self.activation(msg)
        # msg = self.sum_weight[0] * msg + self.sum_weight[1] * node_feature
        # if self.transform is not None:
        #     msg = self.transform(msg)
        # return msg


class node_attention(nn.Module):
    def __init__(self,hidden_dim) -> None:
        super().__init__()
        self.q_linear=nn.Linear(hidden_dim,hidden_dim)
        self.k_linear=nn.Linear(hidden_dim,hidden_dim)
    def forward(self,node_feature:torch.Tensor):
        q=self.q_linear(node_feature)
        k=self.k_linear(node_feature)
        attention=torch.softmax(torch.matmul(q,k.T),dim=1)
        output=torch.matmul(attention,node_feature)
        return output,attention
