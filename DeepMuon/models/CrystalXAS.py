'''
Author: airscker
Date: 2023-09-10 17:32:44
LastEditors: airscker
LastEditTime: 2023-09-16 21:22:27
Description: NULL

Copyright (C) 2023 by Deep Graph Library, All Rights Reserved. 
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import torch
import dgl
import torch.nn.functional as F
from torch import nn
from .base import MLPBlock, ResidualUnit


class GINConv(nn.Module):
    """
    ## Graph Isomorphism Network layer from `How Powerful are Graph Neural Networks?` (https://arxiv.org/pdf/1810.00826.pdf).

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula, default: None.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``), default: 'sum'.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = torch.ones(6, 10)
    >>> lin = torch.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)

    >>> # With activation
    >>> from torch.nn.functional import relu
    >>> conv = GINConv(lin, 'max', activation=relu)
    >>> res = conv(g, feat)
    >>> res
    tensor([[5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [2.5011, 0.0000, 0.0089, 2.0541, 0.8262, 0.0000, 0.0000, 0.1371, 0.0000,
             0.0000]], grad_fn=<ReluBackward0>)
    """

    def __init__(
        self,
        apply_func=None,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(
                "Aggregator type {} not recognized.".format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        _reducer = getattr(dgl.function, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = dgl.function.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = dgl.function.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = dgl.utils.expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, _reducer("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


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
                 gnn_hidden_dims: list[int] = [128, 512],
                 feat_dim: int = 6,
                 prompt_dim: int = 2,
                 mlp_hidden_dims: list = [1024, 512],
                 mlp_dropout=0,
                 xas_type: str = 'XANES'):
        super().__init__()
        xas_types = ['XANES', 'EXAFS', 'XAFS']
        assert xas_type in xas_types, f"'xas_type' must be in {xas_types}, but {xas_type} was given."
        self.xas_type = xas_type
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
                atom_features = self.GIN[i](graph, atom_features)
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
