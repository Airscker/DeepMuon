# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Author: airscker
Date: 2023-09-02 20:55:11
LastEditors: airscker
LastEditTime: 2023-09-16 21:24:23
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import torch
import numpy as np
from torch import nn
from typing import Optional, Sequence, Tuple, Union
from .ADNBlock import ADN

# from monai.networks.blocks import ADN

CONV=(nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d)
CONVTRANS=(nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]
def stride_minus_kernel_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]
class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
          (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        activation: nn.Module=nn.PReLU,
        normalization: nn.Module=nn.InstanceNorm3d,
        dropout_rate: Union[float,int]=0,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type=CONVTRANS[self.spatial_dims-1] if is_transposed else CONV[self.spatial_dims-1]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        self.add_module("ADN",
                        ADN(dim=out_channels,
                            mode=adn_ordering,
                            normalization=normalization,
                            norm_kwargs={},
                            activation=activation,
                            dropout_rate=dropout_rate,
                            dropout_inplace=False))


class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        ResidualUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        strides: Union[Sequence[int], int] = 1,
        adn_ordering: str = "NDA",
        activation: nn.Module=nn.PReLU,
        normalization: nn.Module=nn.InstanceNorm1d,
        dropout_rate: Union[float,int] = 0,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            # conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                activation=activation,
                normalization=normalization,
                dropout_rate=dropout_rate,
                dilation=dilation,
                bias=bias,
                padding=padding,
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = CONV[self.spatial_dims-1]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
