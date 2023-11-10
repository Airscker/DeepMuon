'''
Author: airscker
Date: 2023-10-26 23:09:22
LastEditors: airscker
LastEditTime: 2023-11-10 00:41:49
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch
import torch.nn.functional as F

from torch import nn
from dgl.nn.pytorch import GraphConv,GINConv,GINEConv
from typing import Union
from dgl.utils import expand_as_pair
from DeepMuon.models import MLPBlock,Fourier,RadialBessel
from DeepMuon.dataset import smiles_to_atom_bond_graph

class AtomicNumEmbedding(nn.Module):
    def __init__(self, num_embeddings: int = 100, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embed=nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, x):
        return self.embed(x)

class GatedMLP(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,hidden_dim:list,dropout:float) -> None:
        super().__init__()
        self.gateway=nn.Sequential(
            MLPBlock(in_dim,out_dim,hidden_dim,dropout_rate=dropout),
            nn.Sigmoid(),
        )
        self.output=nn.Sequential(
            MLPBlock(in_dim,out_dim,hidden_dim,dropout_rate=dropout),
            nn.SiLU(),
        )
    def forward(self,x:torch.Tensor):
        return self.output(x)*self.gateway(x)

class AtomConv(nn.Module):
    def __init__(self,
                 node_dim:int,
                 edge_dim:int,
                 hidden_dim:list,
                 dropout:float,
                 encode_self:bool=True) -> None:
        super().__init__()
        self.encode_self=encode_self
        self.gated_MLP=GatedMLP(in_dim=2*node_dim+edge_dim,
                                out_dim=node_dim,
                                hidden_dim=hidden_dim,
                                dropout=dropout)
        self.linear=nn.Linear(node_dim,node_dim)
        self.message=None
    def _message_func(self,edges):
        src_node_feat=edges.src['_node_feat']
        dst_node_feat=edges.dst['_node_feat']
        edge_feat=edges.data['_edge_feat']
        edge_weight=edges.data['_edge_weight']
        res=torch.cat([src_node_feat,dst_node_feat,edge_feat],dim=1)
        res=self.gated_MLP(res)
        self.message=res
        res=res*edge_weight
        return {'m':res}
    def _reduce_func(self,nodes):
        return {'h':nodes.mailbox['m'].sum(dim=1)}
    def forward(self,
                graph:dgl.DGLGraph,
                node_feat:torch.Tensor,
                edge_feat:torch.Tensor,
                edge_weight:torch.Tensor=None):
        '''
        ## Args:
            - graph: dgl.DGLGraph, the atom_graph
            - node_feat: torch.Tensor, shape=(N_a, node_dim)
            - edge_feat: torch.Tensor, shape=(E_a, edge_dim)
            - edge_weight: torch.Tensor, shape=(E_a, node_dim)
        ## Returns:
            - node_feat: torch.Tensor, shape=(N_a, node_dim)
        '''
        with graph.local_scope():
            feat_src,feat_dst=expand_as_pair(node_feat,graph)
            graph.edata['_edge_feat']=edge_feat
            graph.edata['_edge_weight']=edge_weight
            graph.ndata['_node_feat']=feat_src
            graph.update_all(self._message_func,self._reduce_func)
            node_feat=self.linear(graph.ndata['h'])
            if self.encode_self:
                node_feat=feat_dst+node_feat
            return node_feat

class BondConv(nn.Module):
    def __init__(self,
                 node_dim:int,
                 edge_dim:int,
                 atom_embedding_dim:int,
                 hidden_dim:list,
                 dropout:float,
                 encode_self:bool=True) -> None:
        super().__init__()
        self.gated_MLP=GatedMLP(in_dim=2*node_dim+edge_dim+atom_embedding_dim,
                                out_dim=edge_dim,
                                hidden_dim=hidden_dim,
                                dropout=dropout)
        self.linear=nn.Linear(node_dim,node_dim)
        self.atom_feat=None
        self.encode_self=encode_self
    def _message_func(self,edges):
        src_node_feat=edges.src['_node_feat']
        dst_node_feat=edges.dst['_node_feat']
        edge_feat=edges.data['_edge_feat']
        edge_index=edges.data['_edge_index']
        src_node_weight=edges.src['_node_weight']
        dst_node_weight=edges.dst['_node_weight']
        angle_vertex=edge_index[:,1]
        vertex_feat=self.atom_feat[angle_vertex]
        res=torch.cat([src_node_feat,dst_node_feat,edge_feat,vertex_feat],dim=1)
        res=self.gated_MLP(res)
        res=res*src_node_weight*dst_node_weight
        return {'m':res}
    def _reduce_func(self,nodes):
        return {'h':nodes.mailbox['m'].sum(dim=1)}
    def forward(self,
                graph:dgl.DGLGraph,
                node_feat:torch.Tensor,
                edge_feat:torch.Tensor,
                node_weight:torch.Tensor=None,
                edge_index:torch.Tensor=None,
                atom_feat:torch.Tensor=None):
        '''
        ## Args:
            - graph: dgl.DGLGraph, the bond graph
            - node_feat: torch.Tensor, shape=(N_b, node_dim)
            - edge_feat: torch.Tensor, shape=(E_b, edge_dim)
            - node_weight: torch.Tensor, shape=(N_b, edge_dim)
            - edge_index: torch.Tensor, shape=(E_b, 3)
            - atom_feat: torch.Tensor, shape=(N_a, atom_embedding_dim)
        ## Returns:
            - node_feat: torch.Tensor, shape=(N_b, node_dim)
        '''
        self.atom_feat=atom_feat
        with graph.local_scope():
            feat_src,feat_dst=expand_as_pair(node_feat,graph)
            graph.edata['_edge_feat']=edge_feat
            graph.edata['_edge_index']=edge_index
            graph.ndata['_node_feat']=feat_src
            graph.ndata['_node_weight']=node_weight
            graph.update_all(self._message_func,self._reduce_func)
            node_feat=self.linear(graph.ndata['h'])
            if self.encode_self:
                node_feat=feat_dst+node_feat
            return node_feat

class AngleUpdate(nn.Module):
    def __init__(self,
                 bond_dim:int,
                 angle_dim:int,
                 atom_embedding_dim:int,
                 hidden_dim:list,
                 dropout:float,
                 encode_self:bool=True) -> None:
        super().__init__()
        self.gated_MLP=GatedMLP(in_dim=2*bond_dim+angle_dim+atom_embedding_dim,
                                out_dim=angle_dim,
                                hidden_dim=hidden_dim,
                                dropout=dropout)
        self.encode_self=encode_self
    def forward(self,
                atom_feat:torch.Tensor,
                bond_feat:torch.Tensor,
                angle_feat:torch.Tensor,
                angle_index:torch.Tensor,
                bond_graph:dgl.DGLGraph,):
        graph_edges=bond_graph.edges()
        src_bond_feat=bond_feat[graph_edges[0]]
        dst_bond_feat=bond_feat[graph_edges[1]]
        angle_vertex=angle_index[:,1]
        vertex_feat=atom_feat[angle_vertex]
        res=torch.cat([src_bond_feat,dst_bond_feat,angle_feat,vertex_feat],dim=1)
        res=self.gated_MLP(res)
        if self.encode_self:
            res=angle_feat+res
        return res
class MolInteraction(nn.Module):
    def __init__(self,
                 atom_embedding_dim:int=128,
                 bond_embedding_dim:int=128,
                 angle_embedding_dim:int=128,
                 atomconv_hidden_dim:list=[256],
                 atomconv_dropout:float=0,
                 bondconv_hidden_dim:list=[256],
                 bondconv_dropout:float=0,
                 angleconv_hidden_dim:list=[256],
                 angleconv_dropout:float=0,) -> None:
        super().__init__()
        '''
        ## Molucule Interaction Bolck

        ### Args:
            - atom_embedding_dim: int, default=128, the dimension of embedded atomic number.
            - bond_embedding_dim: int, default=128, the dimension of embedded bond length.
            - angle_embedding_dim: int, default=128, the dimension of embedded angle degree.
            - atomconv_hidden_dim: list, default=[256], the hidden dimension of atom convolution.
            - atomconv_dropout: float, default=0, the dropout rate of atom convolution.
            - bondconv_hidden_dim: list, default=[256], the hidden dimension of bond convolution.
            - bondconv_dropout: float, default=0, the dropout rate of bond convolution.
            - angleconv_hidden_dim: list, default=[256], the hidden dimension of angle convolution.
            - angleconv_dropout: float, default=0, the dropout rate of angle convolution.
        
        ### Returns:
            - atomic_num_embedding: torch.Tensor, shape=(N_a, atom_embedding_dim)
            - bond_length_embedding: torch.Tensor, shape=(E_a, bond_embedding_dim)=(N_b, bond_embedding_dim)
            - angle_deg_embedding: torch.Tensor, shape=(E_b, angle_embedding_dim)
        '''
        self.atom_conv=AtomConv(node_dim=atom_embedding_dim,
                                edge_dim=bond_embedding_dim,
                                hidden_dim=atomconv_hidden_dim,
                                dropout=atomconv_dropout,
                                encode_self=True)
        self.bond_conv=BondConv(node_dim=atom_embedding_dim,
                                edge_dim=bond_embedding_dim,
                                atom_embedding_dim=atom_embedding_dim,
                                hidden_dim=bondconv_hidden_dim,
                                dropout=bondconv_dropout,
                                encode_self=True)
        self.angle_update=AngleUpdate(bond_dim=bond_embedding_dim,
                                      angle_dim=angle_embedding_dim,
                                      atom_embedding_dim=atom_embedding_dim,
                                      hidden_dim=angleconv_hidden_dim,
                                      dropout=angleconv_dropout,
                                      encode_self=True)
    def forward(self,
                atom_graph:dgl.DGLGraph,
                bond_graph:dgl.DGLGraph,
                angle_index:torch.Tensor,
                atomic_num_embedding:torch.Tensor,
                bond_length_embedding:torch.Tensor,
                angle_deg_embedding:torch.Tensor,
                atomconv_edge_weight:torch.Tensor,
                bondconv_edge_weight:torch.Tensor):
        # graph=atom_graph.to(device)
        atomic_num_embedding=self.atom_conv(atom_graph,
                                            atomic_num_embedding,
                                            bond_length_embedding,
                                            atomconv_edge_weight)

        # graph=bond_graph.to(device)
        bond_length_embedding=self.bond_conv(bond_graph,
                                             bond_length_embedding,
                                             angle_deg_embedding,
                                             bondconv_edge_weight,
                                             angle_index,
                                             atomic_num_embedding)
        angle_deg_embedding=self.angle_update(atomic_num_embedding,
                                              bond_length_embedding,
                                              angle_deg_embedding,
                                              angle_index,
                                              bond_graph)
        return atomic_num_embedding,bond_length_embedding,angle_deg_embedding
class AngleEncoder(nn.Module):
    def __init__(self,
                 num_angular:int=9,
                 learnable:bool=True) -> None:
        super().__init__()
        '''
        ## Encode an angle given the two bond vectors using Fourier Expansion.

        ### Args:
            - num_angular: int, default=9, number of angular basis to use. Must be an odd integer.
            - learnable: bool, default=True, whether to set the frequencies as learnable parameters.
        ### Returns:
            - angle_embedding: torch.Tensor, shape=(E_b, num_angular)
        '''
        if num_angular%2==0:
            raise ValueError('num_angular must be odd integers')
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion=Fourier(
            order=circular_harmonics_order,
            learnable=learnable,
        )
    def forward(self,angle:torch.Tensor,device:Union[str,torch.device]='cpu'):
        '''
        angle.shape: (E_b, 1)=(N_a, 1)
        '''
        angle=angle.to(device)
        return self.fourier_expansion(angle.squeeze(-1))
class BondLengthEncoder(nn.Module):
    def __init__(self,
                 cutoff: float = 5,
                 num_radial: int = 9,
                 smooth_cutoff: int = 5,
                 learnable: bool = True,) -> None:
        super().__init__()
        '''
        ## Encode a chemical bond given the position of two atoms using Gaussian Distance.

        ### Args:
            - cutoff: float, default=5, the cutoff distance for graph.
            - num_radial: int, default=9, the number of radial basis to use.
            - smooth_cutoff: int, default=5, the smooth cutoff coefficient.
            - learnable: bool, default=False, whether to set the frequencies as learnable parameters.
        
        ### Returns:
            - bond_length_embedding: torch.Tensor, shape=(E_a, num_radial)=(N_b, num_radial)
        '''
        self.rbf_expansion = RadialBessel(
            num_radial=num_radial,
            cutoff=cutoff,
            smooth_cutoff=smooth_cutoff,
            learnable=learnable,
        )
    def forward(self,x:torch.Tensor,device:Union[str,torch.device]='cpu'):
        '''
        x.shape: (E_a, 1)=(N_b, 1)
        '''
        x=x.to(device)
        return self.rbf_expansion(x.squeeze(-1))

class MolSpaceGNN(nn.Module):
    def __init__(self,
                 depth:int=3,
                 atom_dim:int=1,
                 num_angular:int=9,
                 num_radial:int=9,
                 cutoff:float=5,
                 smooth_cutoff:int=5,
                 learnable_rbf:bool=True,
                 densenet:bool=False,
                 residual:bool=False,
                 atom_embedding_dim:int=128,
                 bond_embedding_dim:int=128,
                 angle_embedding_dim:int=128,
                 atom_num_embedding:int=100,
                 atomconv_hidden_dim:list=[256],
                 atomconv_dropout:float=0,
                 bondconv_hidden_dim:list=[256],
                 bondconv_dropout:float=0,
                 angleconv_hidden_dim:list=[256],
                 angleconv_dropout:float=0,) -> None:
        super().__init__()
        '''
        ## Molucule Space For Intra-Molecular Interaction

        ### Args:
            - depth: int, default=3, the depth of interaction blocks.
            - atom_dim: int, default=1, the dimension of the node feature of atom graph.
            - num_angular: int, default=9, number of angular basis for fourier expansion to use. Must be an odd integer.
            - num_radial: int, default=9, the number of radial basis for radial bessel expansion to use.
            - cutoff: float, default=5, the cutoff distance for graph, will be used in radial bessel expansion.
            - smooth_cutoff: int, default=5, the smooth cutoff coefficient, will be used in radial bessel expansion.
            - learnable_rbf: bool, default=True, whether to set the frequencies of fourier/rbf expansion as learnable parameters.
            - densenet: bool, default=False, whether to densely sum the output of each interaction block.
            - residual: bool, default=False, whether to use residual connection in each interaction block.
            - atom_embedding_dim: int, default=128, the dimension of embedded atomic number.
            - bond_embedding_dim: int, default=128, the dimension of embedded bond length.
            - angle_embedding_dim: int, default=128, the dimension of embedded angle degree.
            - atom_num_embedding: int, default=100, the size of the atomic number embedding set.
            - atomconv_hidden_dim: list, default=[256], the hidden dimension of atom convolution.
            - atomconv_dropout: float, default=0, the dropout rate of atom convolution.
            - bondconv_hidden_dim: list, default=[256], the hidden dimension of bond convolution.
            - bondconv_dropout: float, default=0, the dropout rate of bond convolution.
            - angleconv_hidden_dim: list, default=[256], the hidden dimension of angle convolution.
            - angleconv_dropout: float, default=0, the dropout rate of angle convolution.
            
        '''
        if densenet and residual:
            raise ValueError('densenet and residual cannot be True at the same time')
        self.densenet=densenet
        self.residual=residual
        self.bond_length_rbf_encoder=BondLengthEncoder(cutoff=cutoff,
                                                       num_radial=num_radial,
                                                       smooth_cutoff=smooth_cutoff,
                                                       learnable=learnable_rbf)
        self.angle_rbf_encoder=AngleEncoder(num_angular=num_angular,
                                            learnable=learnable_rbf)
        # self.atom_embedding=AtomicNumEmbedding(num_embeddings=atom_num_embedding,
        #                                        embedding_dim=atom_embedding_dim)
        self.atom_embedding=nn.Linear(atom_dim,atom_embedding_dim,bias=False)
        self.bond_embedding=nn.Linear(num_radial,bond_embedding_dim,bias=False)
        self.angle_embedding=nn.Linear(num_angular,angle_embedding_dim,bias=False)
        self.bond_weighting_atomconv=nn.Sequential(
            nn.Linear(num_radial,atom_embedding_dim,bias=False),
            nn.Sigmoid(),
        )
        self.bond_weighting_bondconv=nn.Sequential(
            nn.Linear(num_radial,bond_embedding_dim,bias=False),
            nn.Sigmoid(),
        )
        self.interaction_blocks=nn.ModuleList([
            MolInteraction(atom_embedding_dim=atom_embedding_dim,
                           bond_embedding_dim=bond_embedding_dim,
                           angle_embedding_dim=angle_embedding_dim,
                           atomconv_hidden_dim=atomconv_hidden_dim,
                           atomconv_dropout=atomconv_dropout,
                           bondconv_hidden_dim=bondconv_hidden_dim,
                           bondconv_dropout=bondconv_dropout,
                           angleconv_hidden_dim=angleconv_hidden_dim,
                           angleconv_dropout=angleconv_dropout)
            for _ in range(depth)
        ])
    def _conv_forward(self,
                      atom_graph:dgl.DGLGraph,
                      bond_graph:dgl.DGLGraph,
                      device:Union[str,torch.device]='cpu'):
        '''
        ### Args:
            - atom_graph: dgl.DGLGraph, the atom graph with node feature `atomic_num` and edge feature `bond_length`:
                - atomic_num.shape=(N_a, )
                - bond_length.shape=(E_a, )=(N_b, )
            - bond_graph: dgl.DGLGraph, the bond graph with edge feature `bond_angle` and `angle_index`, also node feature `bond_length`:
                - bond_angle.shape=(E_b, )
                - angle_index.shape=(E_b, 3)
                - bond_length.shape=(N_b, )=(E_a, )
        ### Tips:
            - bond_length/angle encoding: Although bond_length can be divided into identical parts halfly
                (which means we may need to calculate only half of the bond_length when training model),
                but we don't have to need to split it when encoding because encoding has no influence on the identity.
        '''
        atom_graph=atom_graph.to(device)
        '''
        atomic_num.shape=(N_a, )
        bond_length.shape=(E_a, )=(N_b, )
        '''
        atomic_num=atom_graph.ndata['atomic_num']
        bond_length=atom_graph.edata['bond_length']
        bond_length=bond_length.unsqueeze(-1)
        bond_length=self.bond_length_rbf_encoder(bond_length,device)
        atomic_num_embedding=self.atom_embedding(atomic_num)
        bond_length_embedding=self.bond_embedding(bond_length)
        atomconv_edge_weight=self.bond_weighting_atomconv(bond_length)
        bondconv_edge_weight=self.bond_weighting_bondconv(bond_length)

        bond_graph=bond_graph.to(device)
        '''
        bond_angle.shape=(E_b, )
        angle_index.shape=(E_b, 3)
        '''
        bond_angle=bond_graph.edata['bond_angle']
        bond_angle=bond_angle.unsqueeze(-1)
        bond_angle=self.angle_rbf_encoder(bond_angle,device)
        angle_index=bond_graph.edata['angle_index']
        angle_deg_embedding=self.angle_embedding(bond_angle)

        atom_dense=atomic_num_embedding
        bond_dense=bond_length_embedding
        angle_dense=angle_deg_embedding
        for i in range(len(self.interaction_blocks)):
            atomic_num_embedding,bond_length_embedding,angle_deg_embedding=self.interaction_blocks[i](
                atom_graph=atom_graph,
                bond_graph=bond_graph,
                angle_index=angle_index,
                atomic_num_embedding=atomic_num_embedding,
                bond_length_embedding=bond_length_embedding,
                angle_deg_embedding=angle_deg_embedding,
                atomconv_edge_weight=atomconv_edge_weight,
                bondconv_edge_weight=bondconv_edge_weight,
            )
            if self.residual:
                atomic_num_embedding=atomic_num_embedding+atom_dense
                bond_length_embedding=bond_length_embedding+bond_dense
                angle_deg_embedding=angle_deg_embedding+angle_dense
                atom_dense=atomic_num_embedding
                bond_dense=bond_length_embedding
                angle_dense=angle_deg_embedding
            elif self.densenet:
                atom_dense=atom_dense+atomic_num_embedding
                bond_dense=bond_dense+bond_length_embedding
                angle_dense=angle_dense+angle_deg_embedding
        if self.densenet:
            return atom_dense,bond_dense,angle_dense
        else:
            return atomic_num_embedding,bond_length_embedding,angle_deg_embedding
    def _group_conv_forward(self,atom_graph:list[dgl.DGLGraph],bond_graph:list[dgl.DGLGraph],device:Union[str,torch.device]='cpu'):
        '''
        ===========================================================================================================
        ATTENTION: Angle Vertxe Index CANNOT properly match the corresponding graph because of the batch operation.
        ===========================================================================================================

        Because we need to map the angle vertex to the corresponding atom graph to get the atom features by doing `atom_feat[angle_vertex[:,1]]`,
        but batching the graph will make this mapping wrong.
        Which means that as for two graphs with 5 and 6 atoms, the angle vertex indexs of the second graph are still be `0,1,2,3,4,5`,
        which are not the actual index of the batched second graph (they should be `5,6,7,8,9,10`).
        So here we do convolution operations for each graph one by one without batching, which is not efficient but correct.
        ONLY IF we don't have to map the angle vertex to the atom graph, we can do batching, such as suming node features directly of the atom graph in downstream tasks.
        '''
        batch_size=len(atom_graph)
        atomic_embedding_list=[]
        bond_embedding_list=[]
        angle_embedding_list=[]
        for i in range(batch_size):
            atomic_embedding,bond_embedding,angle_embedding=self._conv_forward(atom_graph[i],bond_graph[i],device)
            atomic_embedding_list.append(atomic_embedding)
            bond_embedding_list.append(bond_embedding)
            angle_embedding_list.append(angle_embedding)
        return torch.cat(atomic_embedding_list,dim=0),torch.cat(bond_embedding_list,dim=0),torch.cat(angle_embedding_list,dim=0)
    def forward(self,
                atom_graph:list[dgl.DGLGraph],
                bond_graph:list[dgl.DGLGraph],
                device:Union[str,torch.device]='cpu'):
        return self._group_conv_forward(atom_graph,bond_graph,device)
        # with atom_graph.local_scope():
        #     atom_graph.ndata['atomic_num']=atomic_num_embedding
        #     feat=dgl.mean_nodes(atom_graph,'atomic_num')
        #     return self.mlp(feat).squeeze(-1)
        # print(atom_graph.ndata['atomic_num'])
        # return atomic_num_embedding,bond_length_embedding,angle_deg_embedding

class MulMolSpace(nn.Module):
    def __init__(self,
                 classes:int=1,
                 add_dim:int=0,
                 mlp_dims:list=[],
                 depth:int=3,
                 atom_dim:int=1,
                 num_angular:int=9,
                 num_radial:int=9,
                 cutoff:float=5,
                 smooth_cutoff:int=5,
                 learnable_rbf:bool=True,
                 atom_embedding_dim:int=128,
                 bond_embedding_dim:int=128,
                 angle_embedding_dim:int=128,
                 atom_num_embedding:int=100,
                 atomconv_hidden_dim:list=[256],
                 atomconv_dropout:float=0,
                 bondconv_hidden_dim:list=[256],
                 bondconv_dropout:float=0,
                 angleconv_hidden_dim:list=[256],
                 angleconv_dropout:float=0,) -> None:
        super().__init__()
        self.gnn=MolSpaceGNN(depth=depth,
                             atom_dim=atom_dim,
                             num_angular=num_angular,
                             num_radial=num_radial,
                             cutoff=cutoff,
                             smooth_cutoff=smooth_cutoff,
                             learnable_rbf=learnable_rbf,
                             atom_embedding_dim=atom_embedding_dim,
                             bond_embedding_dim=bond_embedding_dim,
                             angle_embedding_dim=angle_embedding_dim,
                             atom_num_embedding=atom_num_embedding,
                             atomconv_hidden_dim=atomconv_hidden_dim,
                             atomconv_dropout=atomconv_dropout,
                             bondconv_hidden_dim=bondconv_hidden_dim,
                             bondconv_dropout=bondconv_dropout,
                             angleconv_hidden_dim=angleconv_hidden_dim,
                             angleconv_dropout=angleconv_dropout)
        self.regression=MLPBlock(dim_input=2*atom_embedding_dim+add_dim,
                                 hidden_sizes=mlp_dims,
                                 dim_output=classes,)
        self.add_dim=add_dim
    def _feat_extract(self,atom_graph:dgl.DGLGraph,bond_graph:torch.Tensor,device:Union[str,torch.device]='cpu'):
        with atom_graph.local_scope():
            atom_feat,bond_feat,angle_feat=self.gnn(atom_graph,bond_graph,device)
            atom_graph.ndata['atom_feat']=atom_feat
            atom_feat=dgl.mean_nodes(atom_graph,'atom_feat')
            return atom_feat
    def forward(self,solvdata=None,device=None):
        cation_atom_graph: dgl.DGLGraph = solvdata['cation_atom_graphs'].to(device)
        cation_bond_graph: dgl.DGLGraph = solvdata['cation_bond_graphs'].to(device)
        anion_atom_graph: dgl.DGLGraph = solvdata['anion_atom_graphs'].to(device)
        anion_bond_graph: dgl.DGLGraph = solvdata['anion_bond_graphs'].to(device)
        if self.add_dim > 0:
            add_features = solvdata['add_features'].float().to(device)
            add_features = add_features.squeeze()
        cation_feat = self._feat_extract(cation_atom_graph, cation_bond_graph, device)
        anion_feat = self._feat_extract(anion_atom_graph, anion_bond_graph, device)
        output = self.regression(
            torch.cat([cation_feat, anion_feat, add_features], axis=1))
        return output.squeeze(-1)


class MolProperty(MolSpaceGNN):
    def __init__(self,
                 depth: int = 3,
                 atom_dim: int = 1,
                 num_angular: int = 9,
                 num_radial: int = 9,
                 cutoff: float = 5,
                 smooth_cutoff: int = 5,
                 learnable_rbf: bool = True,
                 densenet: bool = False,
                 residual: bool = False,
                 atom_embedding_dim: int = 128,
                 bond_embedding_dim: int = 128,
                 angle_embedding_dim: int = 128,
                 atom_num_embedding: int = 100,
                 atomconv_hidden_dim: list = [256],
                 atomconv_dropout: float = 0,
                 bondconv_hidden_dim: list = [256],
                 bondconv_dropout: float = 0,
                 angleconv_hidden_dim: list = [256],
                 angleconv_dropout: float = 0,
                 mlp_dims: list = [],
                 mlp_out_dim: int = 1) -> None:
        super().__init__(depth,
                         atom_dim,
                         num_angular,
                         num_radial,
                         cutoff,
                         smooth_cutoff,
                         learnable_rbf,
                         densenet,
                         residual,
                         atom_embedding_dim,
                         bond_embedding_dim,
                         angle_embedding_dim,
                         atom_num_embedding,
                         atomconv_hidden_dim,
                         atomconv_dropout,
                         bondconv_hidden_dim,
                         bondconv_dropout,
                         angleconv_hidden_dim,
                         angleconv_dropout)
        self.mlp=MLPBlock(dim_input=atom_embedding_dim,
                          hidden_sizes=mlp_dims,
                          dim_output=mlp_out_dim,)
    def forward(self,atom_graph:list[dgl.DGLGraph],bond_graph:list[torch.Tensor],device:Union[str,torch.device]='cpu'):
        atomic_num_embedding, bond_length_embedding, angle_deg_embedding=self._group_conv_forward(atom_graph,bond_graph,device)
        batched_atom_graphs=dgl.batch(atom_graph)
        batched_atom_graphs=batched_atom_graphs.to(device)
        with batched_atom_graphs.local_scope():
            batched_atom_graphs.ndata['h'] = atomic_num_embedding
            feat = dgl.mean_nodes(batched_atom_graphs, 'h')
            return self.mlp(feat).squeeze(-1)


class TestGNN(nn.Module):

    def __init__(self,
                 depth: int = 3,
                 atom_dim: int = 74,
                 bond_dim: int = 12,
                 densenet: bool = False,
                 residual: bool = False,
                 embedding_dim: int = 128,
                 mlp_dims: list = [],
                 mlp_out_dim: int = 1) -> None:
        super().__init__()
        if densenet and residual:
            raise ValueError('densenet and residual cannot be True at the same time')
        self.densenet = densenet
        self.residual = residual
        self.atom_embedding = nn.Linear(atom_dim,
                                        embedding_dim,
                                        bias=False)
        self.bond_embedding = nn.Linear(bond_dim,
                                        embedding_dim,
                                        bias=False)
        self.interaction_blocks = nn.ModuleList([
            GINEConv(nn.Linear(embedding_dim,
                              embedding_dim),learn_eps=True)
            for _ in range(depth)
        ])
        self.mlp=MLPBlock(dim_input=embedding_dim,
                          hidden_sizes=mlp_dims,
                          dim_output=mlp_out_dim,
                          activation=nn.ReLU)

    def forward(self,
                atom_graph: list[dgl.DGLGraph],
                bond_graph: list[dgl.DGLGraph],
                device: Union[str, torch.device] = 'cpu'):
        atom_graph=dgl.batch(atom_graph)
        bond_graph=dgl.batch(bond_graph)
        atom_graph = atom_graph.to(device)
        atomic_num = atom_graph.ndata['atomic_num']
        bond_length = atom_graph.edata['bond_length']
        atomic_num_embedding = self.atom_embedding(atomic_num)
        bond_length_embedding = self.bond_embedding(bond_length)

        atom_dense = atomic_num_embedding
        with atom_graph.local_scope():
            for i in range(len(self.interaction_blocks)):
                atomic_num_embedding = F.relu(self.interaction_blocks[i](atom_graph,atomic_num_embedding,bond_length_embedding))
                if self.residual:
                    atomic_num_embedding = atomic_num_embedding + atom_dense
                    atom_dense = atomic_num_embedding
                elif self.densenet:
                    atom_dense = atom_dense + atomic_num_embedding
            atom_graph.ndata['h'] = atom_dense
            feat = dgl.mean_nodes(atom_graph, 'h')
            return self.mlp(feat).squeeze(-1)


# smiles='CCOCCC'
# a1,b1=smiles_to_atom_bond_graph(smiles)
# smiles='CCOCC'
# a2,b2=smiles_to_atom_bond_graph(smiles)
# print(a1, '\n', b1.edata)
# print(a2, '\n', b2.edata)
# atomg=[a1,a2]
# bondg=[b1,b2]
# # print(atom_graph.ndata['atomic_num'],atom_graph.edges())
# model=MolProperty(atom_dim=74,depth=1)
# res=model(atomg,bondg)
# # # print(model)
# print(res.shape)
