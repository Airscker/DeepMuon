'''
Author: airscker
Date: 2023-12-11 14:59:57
LastEditors: airscker
LastEditTime: 2024-04-20 22:51:45
Description: NULL

Copyright (C) 2023 by matgl(https://github.com/materialsvirtuallab/matgl), All Rights Reserved.
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

from __future__ import annotations

import os
from functools import lru_cache
from math import pi, sqrt

import numpy as np
import sympy
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn

CWD = os.path.dirname(os.path.abspath(__file__))

'''
Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
order l Spherical Bessel function is the `[l, n]` entry.
'''
SPHERICAL_BESSEL_ROOTS = torch.tensor(np.load(os.path.join(CWD, "sbf_roots.npy")),dtype=torch.float32)

def spherical_bessel_jn(n, x, return_all=False):
    '''
    ## Compute the spherical Bessel function of the first kind of order n for a given x using PyTorch.
    This is a manual implementation since PyTorch does not have a built-in function for this.

    ### Args:
        - n (int): The order of the spherical Bessel function.
        - x (torch.Tensor): The input tensor.
        - return_all (bool): Whether to return all the spherical Bessel function values from order 0 to n.

    ### Returns:
        - torch.Tensor: The spherical Bessel function values for each element in x.\\
            If return_all is True, the shape is [n + 1, x.shape[0]], otherwise the shape is [x.shape[0]].
    '''
    assert len(x.shape) == 1, 'x must be a 1D tensor.'
    # Handle the case where x is 0 to avoid division by zero
    result = torch.zeros_like(x)
    x_nonzero = x[x != 0]
    # Compute the spherical Bessel function of the first kind
    # For n = 0
    if n == 0:
        result[x != 0] = torch.sin(x_nonzero) / x_nonzero
    # For n > 0
    else:
        jn_m1 = torch.sin(
            x_nonzero) / x_nonzero  # J_{n-1} term, start with n = 0
        jn = (torch.sin(x_nonzero) - x_nonzero * torch.cos(x_nonzero)
              ) / x_nonzero**2  # J_n term, start with n = 1
        jn_list = [jn_m1, jn]
        for m in range(1, n):
            jn_m1, jn = jn_list[-2:]
            # Recurrence relation: J_{n+1}(x) = (2n + 1) / x * J_n(x) - J_{n-1}(x)
            jn_p1 = ((2 * m + 1) / x_nonzero) * jn - jn_m1
            jn_list.append(jn_p1)
        jn_list = torch.stack(jn_list)
        if return_all:
            result = torch.zeros((n + 1, x.shape[0]))
            result[:, x != 0] = jn_list
        else:
            result[x != 0] = jn_list[-1]
    return result

def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool, device: torch.device):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = torch.tensor([1] * max_l)
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = 2 * torch.arange(max_l) + 1
        repeats_sbf = repeats_sbf.repeat(max_n)
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * torch.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    
    repeats_sbf, block_size = repeats_sbf.to(device), block_size.to(device)
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf,
                                 block_size=block_size,
                                 repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])

@lru_cache(maxsize=128)
def spherical_bessel_roots(max_l: int, max_n: int):
    """Calculate the spherical Bessel roots. The n-th root of the l-th
    spherical bessel function is the `[l, n]` entry of the return matrix.
    The calculation is based on the fact that the n-root for l-th
    spherical Bessel function `j_l`, i.e., `z_{j, n}` is in the range
    `[z_{j-1,n}, z_{j-1, n+1}]`. On the other hand we know precisely the
    roots for j0, i.e., sinc(x).

    Args:
        max_l: max order of spherical bessel function
        max_n: max number of roots
    Returns: root matrix of size [max_l, max_n]
    """
    temp_zeros = np.arange(1, max_l + max_n + 1) * pi  # j0
    roots = [temp_zeros[:max_n]]
    for i in range(1, max_l):
        roots_temp = []
        for j in range(max_n + max_l - i):
            low = temp_zeros[j]
            high = temp_zeros[j + 1]
            root = brentq(lambda x, v: spherical_jn(v, x), low, high, (i, ))
            roots_temp.append(root)
        temp_zeros = np.array(roots_temp)
        roots.append(temp_zeros[:max_n])
    return np.array(roots)


def _block_repeat(array, block_size, repeats):
    col_index = torch.arange(array.size()[1]).to(array.device)
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start:start + b], [repeats[i]]))
        start += b
    indices = torch.cat(indices, axis=0)
    return torch.index_select(array, 1, indices)


@lru_cache(maxsize=128)
def _get_lambda_func(max_n, cutoff: float = 5.0):
    r = sympy.symbols("r")
    en = [i**2 * (i + 2)**2 / (4 * (i + 1)**4 + 1) for i in range(max_n)]

    dn = [1.0]
    for i in range(1, max_n):
        dn_value = 1 - en[i] / dn[-1]
        dn.append(dn_value)

    fnr = [
        (-1)**i * sqrt(2.0) * pi / cutoff**1.5 * (i + 1) * (i + 2) /
        sympy.sqrt(1.0 * (i + 1)**2 + (i + 2)**2) *
        (sympy.sin(r * (i + 1) * pi / cutoff) /
         (r * (i + 1) * pi / cutoff) + sympy.sin(r * (i + 2) * pi / cutoff) /
         (r * (i + 2) * pi / cutoff)) for i in range(max_n)
    ]

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr_value = 1 / sympy.sqrt(
            dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
        gnr.append(gnr_value)
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]


def get_segment_indices_from_n(ns):
    """Get segment indices from number array. For example if
    ns = [2, 3], then the function will return [0, 0, 1, 1, 1].

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns:
        torch.Tensor: segment indices tensor
    """
    segments = torch.zeros(ns.sum(), dtype=torch.int64)
    segments[ns.cumsum(0)[:-1]] = 1
    return segments.cumsum(0)


def get_range_indices_from_n(ns):
    """Give ns = [2, 3], return [0, 1, 0, 1, 2].

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns: range indices
    """
    max_n = torch.max(ns)
    n = ns.size(dim=0)
    n_range = torch.arange(max_n)
    matrix = n_range.tile([n, 1], )
    mask = torch.arange(max_n)[None, :] < ns[:, None]

    #    return matrix[mask]
    return torch.masked_select(matrix, mask)


def repeat_with_n(ns, n):
    """Repeat the first dimension according to n array.

    Args:
        ns (torch.tensor): tensor
        n (torch.tensor): a list of replications

    Returns: repeated tensor

    """
    return torch.repeat_interleave(ns, n, dim=0)


def broadcast_states_to_bonds(g, state_feat):
    """Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate].

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes
    """
    return state_feat.repeat((g.num_edges(), 1))


def broadcast_states_to_atoms(g, state_feat):
    """Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate].

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes

    """
    return state_feat.repeat((g.num_nodes(), 1))


def scatter_sum(input_tensor: torch.Tensor, segment_ids: torch.Tensor,
                num_segments: int, dim: int) -> torch.Tensor:
    """Scatter sum operation along the specified dimension. Modified from the
    torch_scatter library (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor (torch.Tensor): The input tensor to be scattered.
        segment_ids (torch.Tensor): Segment ID for each element in the input tensor.
        num_segments (int): The number of segments.
        dim (int): The dimension along which the scatter sum operation is performed (default: -1).

    Returns:
        resulting tensor
    """
    segment_ids = broadcast(segment_ids, input_tensor, dim)
    size = list(input_tensor.size())
    if segment_ids.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = num_segments
    output = torch.zeros(size, dtype=input_tensor.dtype)
    return output.scatter_add_(dim, segment_ids, input_tensor)


def unsorted_segment_fraction(data: torch.Tensor, segment_ids: torch.Tensor,
                              num_segments: int):
    """Segment fraction
    Args:
        data (torch.tensor): original data
        segment_ids (torch.tensor): segment ids
        num_segments (int): number of segments
    Returns:
        data (torch.tensor): data after fraction.
    """
    segment_sum = scatter_sum(input_tensor=data,
                              segment_ids=segment_ids,
                              dim=0,
                              num_segments=num_segments)
    sums = torch.gather(segment_sum, 0, segment_ids)
    return torch.div(data, sums)


def broadcast(input_tensor: torch.Tensor, target_tensor: torch.Tensor,
              dim: int):
    """Broadcast input tensor along a given dimension to match the shape of the target tensor.
    Modified from torch_scatter library (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor: The tensor to broadcast.
        target_tensor: The tensor whose shape to match.
        dim: The dimension along which to broadcast.

    Returns:
        resulting input tensor after broadcasting
    """
    if input_tensor.dim() == 1:
        for _ in range(dim):
            input_tensor = input_tensor.unsqueeze(0)
    for _ in range(input_tensor.dim(), target_tensor.dim()):
        input_tensor = input_tensor.unsqueeze(-1)
    target_shape = list(target_tensor.shape)
    target_shape[dim] = input_tensor.shape[dim]
    return input_tensor.expand(target_shape)
