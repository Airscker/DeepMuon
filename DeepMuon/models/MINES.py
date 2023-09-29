'''
Author: airscker
Date: 2023-05-23 14:36:30
LastEditors: airscker
LastEditTime: 2023-09-27 16:30:27
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
    def forward(self,graph:dgl.DGLGraph,node_feature:torch.Tensor):
        with graph.local_scope():
            output=self.gcn1(graph,node_feature)
            output=self.gcn2(graph,F.relu(output))
            # return node_feature+F.relu(output)
            return F.relu(output)

class SolvGNNV3(nn.Module):
    def __init__(self, in_dim=74, hidden_dim=256, add_dim=0, mlp_dims=[1024,512],dropout=0,
                 gcr_layers=5 ,n_classes=1, res_connection=False,allow_zero_in_degree=True,freeze_GNN=False) -> None:
        super().__init__()
        self.add_dim=add_dim
        self.res_connection=res_connection
        self.freeze_GNN=freeze_GNN
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
            for i in range(len(self.node_gcr)):
                feature=self.node_gcr[i](graph,node_feature)
                if self.res_connection and i>0:
                    node_feature=node_feature+feature
                else:
                    node_feature=feature
            graph.ndata['h']=node_feature
            node_mean=dgl.mean_nodes(graph,'h')
            if self.add_dim>0:
                node_mean=torch.cat([node_mean,add_features],axis=1)
            output=self.regression(node_mean)
            return output.squeeze(-1)
    def freeze_GNNPart(self):
        if self.freeze_GNN:
            for para in self.node_gcr.parameters():
                para.requires_grad=False
            for para in self.gcn.parameters():
                para.requires_grad=False
    def train(self,mode=True):
        super().train(mode)
        self.freeze_GNNPart()



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
                 mlp_dims=[2048,1024,512],
                 dropout_rate=0,
                 norm=True,
                 gcr_layers=25,
                 n_classes=1,
                 res_connection=False,
                 allow_zero_in_degree=True,
                 freeze_GNN=False) -> None:
        super().__init__()
        self.res_connection=res_connection
        gnn_hidden_dims=[in_dim]+[hidden_dim]*gcr_layers
        self.node_update=nn.ModuleList([
            nn.Linear(gnn_hidden_dims[i],gnn_hidden_dims[i+1]) for i in range(len(gnn_hidden_dims)-1)
        ])
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
            for i in range(len(self.gnn)):
                feature=self.gnn[i](graph,graph_ndata)
                if self.res_connection and i>0:
                    graph_ndata=graph_ndata+feature
                else:
                    graph_ndata=feature
            graph.ndata['h']=graph_ndata
            node_mean=dgl.mean_nodes(graph,'h')
            if self.add_dim>0:
                node_mean=torch.cat([node_mean,add_features],axis=1)
            output=self.regression(node_mean)
            return output.squeeze(-1)