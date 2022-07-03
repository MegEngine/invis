#!/usr/bin/env python3

# some code of this file comes from the follow file:
# https://github.com/pytorch/pytorch/blob/62ca5a81c0afbe35f315b5c6ac33232a674b25cd/torch/nn/modules/upsampling.py
from typing import Iterable

from .. import functional as F
from .module import Module


class Upsample(Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.  # noqa
    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.
    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):  # noqa
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name', 'recompute_scale_factor']  # noqa
    name: str
    size: Iterable[int]
    scale_factor: Iterable[float]
    mode: str
    align_corners: bool
    recompute_scale_factor: bool

    def __init__(
        self, size=None, scale_factor=None, mode='nearest',
        align_corners=None, recompute_scale_factor=None
    ):
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def _module_info_str(self) -> str:
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class UpsamplingNearest2d(Upsample):
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode='nearest')


class UpsamplingBilinear2d(Upsample):

    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor, mode='bilinear', align_corners=True)  # noqa
