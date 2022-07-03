#!/usr/bin/env python3

import megengine.module as M

import invis

from .patch import patch_attribute, patch_method

__all__ = ["MaxPool2d"]


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
