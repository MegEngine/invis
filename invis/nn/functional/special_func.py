#!/usr/bin/env python3

import megengine as mge
import megengine.functional as F

import invis as inv

from .helper import ensure_tensor_type, values_indices

__all__ = [
    "all", "range", "min", "max", "argmin", "argmax",
]


@ensure_tensor_type
def all(input):
    return inv.logical_not(input.astype("bool")).sum() == 0


@ensure_tensor_type
def range(
    start=0, end=None, step=1, *, out=None, dtype="float32", device=None, requires_grad=False
):
    # range function might be deprecated in future
    if end is None:
        start, end = 0, start
    if (end - start) // step * step == end - start:
        return F.arange(start, end + step, step, dtype=dtype)
    else:
        return F.arange(start, end, step, dtype=dtype)


@ensure_tensor_type
def argmin(input, dim=None, keepdim=False):
    return F.argmin(input, axis=dim, keepdims=keepdim)


@ensure_tensor_type
def argmax(input, dim=None, keepdim=False):
    return F.argmax(input, axis=dim, keepdims=keepdim)


def min(input, dim=None, keepdim=False, *, out=None):
    values = mge.Tensor.min(input, axis=dim, keepdims=keepdim)
    values = inv.Tensor(values)
    if dim is None:
        return values
    else:
        indices = argmin(input, dim, keepdim)
        return values_indices(values=values, indices=indices)


def max(input, dim=None, keepdim=False, *, out=None):
    values = mge.Tensor.max(input, axis=dim, keepdims=keepdim)
    values = inv.Tensor(values)
    if dim is None:
        return values
    else:
        indices = argmax(input, dim, keepdim)
        return values_indices(values=values, indices=indices)
