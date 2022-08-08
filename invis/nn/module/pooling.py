#!/usr/bin/env python3

import megengine.module as M

import invis

from ..functional import adaptive_avg_pool1d, adaptive_max_pool1d
from .module import Module
from .patch import patch_attribute, patch_method

__all__ = [
    "MaxPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveAvgPool1d",
]


class MaxPool2d(M.Module):

    __constants__ = ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
    return_indices: bool
    ceil_mode: bool

    def __init__(
        self, kernel_size, stride=None,
        padding=0, dilation=1, return_indices=False, ceil_mode=False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        patch_attribute(self)

    def forward(self, inp):
        return invis.max_pool2d(
            inp, self.kernel_size, self.stride,
            padding=self.padding, ceil_mode=self.ceil_mode,
        )

    def _module_info_string(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


MaxPool2d = patch_method(MaxPool2d, patch_override=False)


# class AvgPool2d(MGE_AvgPool2d):
#     pass


class AdaptiveAvgPool1d(Module):
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.
    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.
    """
    __constants__ = ['output_size']

    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size
        patch_attribute(self)

    def forward(self, input):
        return adaptive_avg_pool1d(input, self.output_size)

    def extra_repr(self) -> str:
        return 'output_size={}'.format(self.output_size)


class AdaptiveMaxPool1d(Module):
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.
    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.
    """
    __constants__ = ['output_size']

    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size
        patch_attribute(self)

    def forward(self, input):
        return adaptive_max_pool1d(input, self.output_size)

    def extra_repr(self) -> str:
        return 'output_size={}'.format(self.output_size)
