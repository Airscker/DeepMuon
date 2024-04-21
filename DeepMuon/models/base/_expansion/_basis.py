'''
Author: airscker
Date: 2023-10-26 23:01:51
LastEditors: airscker
LastEditTime: 2024-04-20 22:58:46
Description: NULL

Copyright (C) 2023 by matgl(https://github.com/materialsvirtuallab/matgl), All Rights Reserved.
Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''

from __future__ import annotations

import torch
import sympy
import numpy as np

from torch import Tensor, nn
from functools import lru_cache
from math import pi, sqrt
from typing import Union, Tuple

from ._maths import _get_lambda_func, SPHERICAL_BESSEL_ROOTS, combine_sbf_shf


class FourierExpansion(nn.Module):
    """
    Fourier Expansion for angle features.
    from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/model/basis.py
    """

    def __init__(self, order: int = 5, learnable: bool = False) -> None:
        """Initialize the Fourier expansion.

        Args:
            order (int): the maximum order, refer to the N in eq 1 in CHGNet paper
                Default = 5
            learnable (bool): whether to set the frequencies as learnable parameters
                Default = False
        """
        super().__init__()
        self.order = order
        # Initialize frequencies at canonical
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.arange(1, order + 1, dtype=torch.float),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies", torch.arange(1, order + 1, dtype=torch.float)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier expansion to a feature Tensor."""
        result = x.new_zeros(x.shape[0], 1 + 2 * self.order)
        result[:, 0] = 1 / torch.sqrt(torch.tensor([2]))
        tmp = torch.outer(x, self.frequencies)
        result[:, 1 : self.order + 1] = torch.sin(tmp)
        result[:, self.order + 1 :] = torch.cos(tmp)
        return result / np.sqrt(np.pi)

class GaussianExpansion(nn.Module):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom.
    """

    def __init__(
        self,
        min: float = 0,
        max: float = 5,
        step: float = 0.5,
        var: float = None,
    ) -> None:
        """Gaussian Expansion
        expand a scalar feature to a soft-one-hot feature vector.

        Args:
            min (float): minimum Gaussian center value
            max (float): maximum Gaussian center value
            step (float): Step size between the Gaussian centers
            var (float): variance in gaussian filter, default to step
        """
        super().__init__()
        assert min < max
        assert max - min > step
        self.register_buffer("gaussian_centers", torch.arange(min, max + step, step))
        self.var = var or step
        if self.var <= 0:
            raise ValueError(f"{var=} must be positive")

    def expand(self, features: Tensor) -> Tensor:
        """Apply Gaussian filter to a feature Tensor.

        Args:
            features (Tensor): tensor of features [n]

        Returns:
            expanded features (Tensor): tensor of Gaussian distances [n, dim]
            where the expanded dimension will be (dmax - dmin) / step + 1
        """
        return torch.exp(
            -((features.reshape(-1, 1) - self.gaussian_centers) ** 2) / self.var**2
        )

class RadialBesselFunction(torch.nn.Module):
    """
    1D Bessel Basis
    from: https://github.com/TUM-DAML/gemnet_pytorch/.
    """

    def __init__(
        self,
        num_radial: int = 9,
        cutoff: float = 5,
        learnable: bool = False,
        smooth_cutoff: int = 5,
    ) -> None:
        """Initialize the SmoothRBF function.

        Args:
            num_radial (int): Controls maximum frequency
                Default = 9
            cutoff (float):  Cutoff distance in Angstrom.
                Default = 5
            learnable (bool): whether to set the frequencies learnable
                Default = False
            smooth_cutoff (int): smooth cutoff strength
                Default = 5
        """
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff)**0.5

        # Initialize frequencies at canonical positions
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.Tensor(
                    np.pi *
                    np.arange(1, self.num_radial + 1, dtype=np.float32)),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies",
                np.pi *
                torch.arange(1, self.num_radial + 1, dtype=torch.float),
            )
        if smooth_cutoff is not None:
            self.smooth_cutoff = CutoffPolynomial(cutoff=cutoff,
                                                  cutoff_coeff=smooth_cutoff)
        else:
            self.smooth_cutoff = None

    def forward(
        self,
        dist: Tensor,
        return_smooth_factor: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Apply Bessel expansion to a feature Tensor.

        Args:
            dist (Tensor): tensor of distances [n, 1]
            return_smooth_factor (bool): whether to return the smooth factor
                Default = False

        Returns:
            out (Tensor): tensor of Bessel distances [n, dim]
            where the expanded dimension will be num_radial
            smooth_factor (Tensor): tensor of smooth factors [n, 1]
        """
        dist = dist[:, None]  # shape = (nEdges, 1)
        d_scaled = dist * self.inv_cutoff
        out = self.norm_const * torch.sin(self.frequencies * d_scaled) / dist
        if self.smooth_cutoff is not None:
            smooth_factor = self.smooth_cutoff(dist)
            out = smooth_factor * out
            if return_smooth_factor:
                return out, smooth_factor
        return out

class CutoffPolynomial(nn.Module):
    """
    Polynomial soft-cutoff function for atom graph
    from: https://github.com/TUM-DAML/gemnet_pytorch/blob/-/gemnet/model/layers/envelope.py.
    """

    def __init__(self, cutoff: float = 5, cutoff_coeff: float = 5) -> None:
        """Initialize the polynomial cutoff function.

        Args:
            cutoff (float): cutoff radius (A) in atom graph construction
            Default = 5
            cutoff_coeff (float): the strength of soft-Cutoff
            0 will disable the cutoff, returning 1 at every r
            for positive numbers > 0, the smaller cutoff_coeff is, the faster this function
                decays. Default = 5.
        """
        super().__init__()
        self.cutoff = cutoff
        self.p = cutoff_coeff
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, r: Tensor) -> Tensor:
        """Polynomial cutoff function.

        Args:
            r (Tensor): radius distance tensor

        Returns:
            polynomial cutoff functions: decaying from 1 at r=0 to 0 at r=cutoff
        """
        if self.p != 0:
            r_scaled = r / self.cutoff
            env_val = (1 + self.a * r_scaled**self.p +
                       self.b * r_scaled**(self.p + 1) +
                       self.c * r_scaled**(self.p + 2))
            return torch.where(r_scaled < 1, env_val,
                               torch.zeros_like(r_scaled))
        return r.new_ones(r.shape)

class SphericalBesselFunction(nn.Module):
    """Calculate the spherical Bessel function based on sympy + pytorch implementations."""

    def __init__(self,
                 max_l: int,
                 max_n: int = 5,
                 cutoff: float = 5.0,
                 smooth: bool = False):
        """Args:
        max_l: int, max order (excluding l)
        max_n: int, max number of roots used in each l
        cutoff: float, cutoff radius
        smooth: Whether to smooth the function.
        """
        super().__init__()
        self.max_l = max_l
        self.max_n = max_n
        self.register_buffer("cutoff", torch.tensor(cutoff).float())
        self.smooth = smooth
        self.register_buffer("SPHERICAL_BESSEL_ROOTS",SPHERICAL_BESSEL_ROOTS)
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """Spherical basis functions based on Rayleigh formula. This function
        generates
        symbolic formula.

        Returns: list of symbolic functions
        """
        x = sympy.symbols("x")
        funcs = [
            sympy.expand_func(sympy.functions.special.bessel.jn(i, x))
            for i in range(self.max_l + 1)
        ]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list:
        return _get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Args:
            r: torch.Tensor, distance tensor, 1D.

        Returns:
            torch.Tensor: [n, max_n * max_l] spherical Bessel function results
        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        r_c = r.clone()
        r_c[r_c > self.cutoff] = self.cutoff
        roots = self.SPHERICAL_BESSEL_ROOTS[:self.max_l, :self.max_n]

        results = []
        factor = sqrt(2.0 / self.cutoff**3)
        for i in range(self.max_l):
            root = roots[i].clone()
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r_c[:, None] * root[None, :] / self.cutoff) * factor /
                torch.abs(func_add1(root[None, :])))
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
        """Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff.

        Args:
            r: torch.Tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis

        Returns:
            basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=torch.float32)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r

class SphericalHarmonicsFunction(nn.Module):
    """Spherical Harmonics function."""

    def __init__(self, max_l: int, use_phi: bool = True):
        """
        Args:
            max_l: int, max l (excluding l)
            use_phi: bool, whether to use the polar angle. If not,
            the function will compute `Y_l^0`.
        """
        super().__init__()
        self.max_l = max_l
        self.use_phi = use_phi
        funcs = []
        theta, phi = sympy.symbols("theta phi")
        for lval in range(self.max_l):
            m_list = range(-lval, lval +
                           1) if self.use_phi else [0]  # type: ignore
            for m in m_list:
                func = sympy.functions.special.spherical_harmonics.Znm(
                    lval, m, theta, phi).expand(func=True)
                funcs.append(func)
        # replace all theta with cos(theta)
        cos_theta = sympy.symbols("costheta")
        funcs = [i.subs({theta: sympy.acos(cos_theta)}) for i in funcs]
        self.orig_funcs = [sympy.simplify(i).evalf() for i in funcs]
        self.funcs = [
            sympy.lambdify([cos_theta, phi], i, [{
                "conjugate": torch.conj
            }, torch]) for i in self.orig_funcs
        ]
        self.funcs[0] = _y00

    def __call__(self, cos_theta, phi=None):
        """Args:
            cos_theta: Cosine of the azimuthal angle
            phi: torch.Tensor, the polar angle.

        Returns: [n, m] spherical harmonic results, where n is the number
            of angles. The column is arranged following
            `[Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^1, Y_2^{-2}, ...]`
        """
        # cos_theta = torch.tensor(cos_theta, dtype=torch.complex64)
        # phi = torch.tensor(phi, dtype=torch.complex64)
        return torch.stack([func(cos_theta, phi) for func in self.funcs],
                           axis=1)
        # results = results.type(dtype=DataType.torch_float)
        # return results

def _y00(theta, phi):
    r"""Spherical Harmonics with `l=m=0`.

    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: torch.Tensor, the azimuthal angle
        phi: torch.Tensor, the polar angle

    Returns: `Y_0^0` results
    """
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)

def spherical_bessel_smooth(r: Tensor,
                            cutoff: float = 5.0,
                            max_n: int = 10) -> Tensor:
    """This is an orthogonal basis with first
    and second derivative at the cutoff
    equals to zero. The function was derived from the order 0 spherical Bessel
    function, and was expanded by the different zero roots.

    Ref:
        https://arxiv.org/pdf/1907.02374.pdf

    Args:
        r: torch.Tensor distance tensor
        cutoff: float, cutoff radius
        max_n: int, max number of basis, expanded by the zero roots

    Returns: expanded spherical harmonics with derivatives smooth at boundary

    """
    n = torch.arange(max_n).type(dtype=torch.float32)[None, :]
    r = r[:, None]
    fnr = ((-1)**n * sqrt(2.0) * pi / cutoff**1.5 * (n + 1) * (n + 2) /
           torch.sqrt(2 * n**2 + 6 * n + 5) *
           (_sinc(r * (n + 1) * pi / cutoff) + _sinc(r *
                                                     (n + 2) * pi / cutoff)))
    en = n**2 * (n + 2)**2 / (4 * (n + 1)**4 + 1)
    dn = [torch.tensor(1.0)]
    for i in range(1, max_n):
        dn_value = 1 - en[0, i] / dn[-1]
        dn.append(dn_value)
    dn = torch.stack(dn)  # type: ignore
    gn = [fnr[:, 0]]
    for i in range(1, max_n):
        gn_value = 1 / torch.sqrt(
            dn[i]) * (fnr[:, i] + torch.sqrt(en[0, i] / dn[i - 1]) * gn[-1])
        gn.append(gn_value)

    return torch.t(torch.stack(gn))


def _sinc(x):
    return torch.sin(x) / x

class SphericalBesselWithHarmonics(nn.Module):
    """Expansion of basis using Spherical Bessel and Harmonics."""

    def __init__(self, max_n: int, max_l: int, cutoff: float, use_smooth: bool,
                 use_phi: bool):
        """
        ## Init SphericalBesselWithHarmonics.

        ### Args:
            - max_n: Degree of radial basis functions.
            - max_l: Degree of angular basis functions.
            - cutoff: Cutoff sphere.
            - use_smooth: Whether using smooth version of SBFs or not.
            - use_phi: Using phi as angular basis functions.
        
        ### Returns:
            Input tensor: [n, ] bond lengths; [n, ] cos(theta); [n, ] phi.
            Output tensor: [n, max_n * max_l**2 ] combined basis functions.
        """
        super().__init__()

        assert max_n <= 64
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi
        self.use_smooth = use_smooth

        # retrieve formulas
        self.shf = SphericalHarmonicsFunction(self.max_l, self.use_phi)
        if self.use_smooth:
            self.sbf = SphericalBesselFunction(self.max_l,
                                               self.max_n * self.max_l,
                                               self.cutoff, self.use_smooth)
        else:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n,
                                               self.cutoff, self.use_smooth)

    def forward(self, bond_lengths: Tensor, cos_theta: Tensor, phi: Tensor, device: Union[str, torch.device]='cpu'):
        '''
        ### Args:
            - bond_lengths: torch.Tensor, shape=(n_edges, 1), bond lengths
            - cos_theta: torch.Tensor, shape=(n_edges, 1), cos(theta)
            - phi: torch.Tensor, shape=(n_edges, 1), phi
        ### Returns:
            - torch.Tensor, shape=(n_edges, max_n * max_l**2), combined basis functions
        '''
        sbf=self.sbf(bond_lengths)
        shf=self.shf(cos_theta, phi)
        # sbf = self.sbf(line_graph.edata["triple_bond_lengths"])
        # shf = self.shf(line_graph.edata["cos_theta"], line_graph.edata["phi"])
        return combine_sbf_shf(sbf,
                               shf,
                               max_n=self.max_n,
                               max_l=self.max_l,
                               use_phi=self.use_phi,
                               device=device)
