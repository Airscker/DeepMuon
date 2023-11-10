'''
Author: airscker
Date: 2023-10-24 22:10:30
LastEditors: airscker
LastEditTime: 2023-10-24 22:10:47
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import dgl
import torch

def CombineGraph(graphs:list[dgl.DGLGraph],add_global:bool=True,bi_direction:bool=True,add_self_loop:bool=True):
    '''
    ## Combine a list of graphs into a single graph

    ### Args:
    - graphs: list of graphs, make sure every node/edge in the graph has the same feature dimension, 
        also, the node/edge feature dimension should be the same as the other graphs in the list.
    - add_global: whether to add a global node to the graph, default is `True`. 
        If enabled, every node of every subgraph in the list will be connected to the global node.
    - bi_direction: whether to add bi-directional edges between the global node (if exists) and every other node, default is `True`. 
        If enabled, the global node will be connected to every other node in the graph, and vice versa.
        If disabled, the every node in the graph will be connected to the global node but not vice versa.
    - add_self_loop: whether to add self-loop to global node in the graph, default is `True` (Self loops of subgraph nodes are not added here).
    '''
    combined_graph = dgl.batch(graphs)
    if add_global:
        combined_graph=dgl.add_nodes(combined_graph,1)
        num_node=combined_graph.num_nodes()-1
        start=[]
        end=[]
        for i in range(num_node):
            start.append(i)
            end.append(num_node)
        if bi_direction:
            for i in range(num_node):
                start.append(num_node)
                end.append(i)
        if add_self_loop:
            start.append(num_node)
            end.append(num_node)
        combined_graph=dgl.add_edges(combined_graph,torch.LongTensor(start),torch.LongTensor(end))
    return combined_graph