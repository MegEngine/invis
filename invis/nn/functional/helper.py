#!/usr/bin/env python3

import functools
from collections import namedtuple

import megengine as mge

import invis as inv

__all__ = [
    "ensure_tensor_type",
    "ensure_tensor_list",
    "inplace",
    "swap_argument",
    "values_indices",
]


# this is a helper function to use like invis.min/max/sort
# e.g. : invis.min(x, dim=2).indices
values_indices = namedtuple("values_indices", ["values", "indices"])


def ensure_tensor_type(f):

    @functools.wraps(f)
    def func(*args, **kwargs):
        ret = f(*args, **kwargs)
        if isinstance(ret, mge.tensor):
            ret = inv.tensor(ret)
        return ret

    return func


def ensure_tensor_list(f):

    @functools.wraps(f)
    def func(*args, **kwargs):
        ret = f(*args, **kwargs)
        if isinstance(ret, (tuple, list)):
            return type(ret)(
                [inv.tensor(v) if isinstance(v, mge.Tensor) else v for v in ret]
            )
        return ret

    return func


def swap_argument(f):
    """
    This wrapper is used to swap the order of the arguments of the function.
    python requires functions such as `__radd__`
    """

    @functools.wraps(f)
    def swap_f(x, y):
        return f(y, x)
    return swap_f


def inplace(f):
    """
    This wrapper is used to make the function inplace.
    python requires functions such as `__iadd__`
    """

    @functools.wraps(f)
    def inplace_f(*args):
        result = f(*args)
        assert isinstance(result, mge.Tensor)
        tensor = args[0]
        tensor._reset(result)
        return tensor

    return inplace_f
