'''
Author: airscker
Date: 2023-05-23 14:36:30
LastEditors: airscker
LastEditTime: 2023-05-23 17:53:31
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, NNConv

class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
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
    def __init__(self, in_dim=74, hidden_dim=256, n_classes=1):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim+1,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_classes)
        )
        
    def forward(self, solvdata,empty_solvsys,device):
        graph=solvdata['graph'].to(device)
        with graph.local_scope():
            graph_ndata=graph.ndata['h'].float()
            inter_hb=solvdata['inter_hb'][:,None].float().to(device)
            be_salt=solvdata['be_salt'][:,None].to(device)
            be_ps=solvdata['be_ps'][:,None].to(device)
            ip=solvdata['ip'][:,None].to(device)
            graph.ndata['h']=F.relu(self.conv2(graph,F.relu(self.conv1(graph,graph_ndata))))
            graph_mean=dgl.mean_nodes(graph,'h')
            graph_mean=torch.cat([graph_mean,inter_hb],axis=1)
            graph_mean=torch.cat([graph_mean,graph_mean])
            inter_feature=torch.cat([inter_hb,be_salt,be_ps,ip],axis=0)
            gh_feature=self.global_conv1(empty_solvsys,graph_mean,inter_feature)
            output=self.regression(gh_feature)
            output = torch.cat((output[0:len(output)//2,:],output[len(output)//2:,:]),axis=1)
            output=torch.mean(output,dim=1).unsqueeze(1)
            return output   
