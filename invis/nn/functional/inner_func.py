#!/usr/bin/env python3
from functools import partial

import numpy as np

from megengine.core.ops.builtin import Elemwise
from megengine.core.tensor.array_method import _elwise

from .helper import ensure_tensor_type

__all__ = [
    "add", "sub", "mul", "true_divide", "floor_divide",
    "fmod", "pow", "bitwise_left_shift", "bitwise_right_shift",
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "eq", "ne", "le", "lt", "ge", "gt",
]


add = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.ADD))
sub = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.SUB))
mul = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.MUL))
true_divide = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.TRUE_DIV))
floor_divide = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.FLOOR_DIV))
fmod = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.MOD))
pow = ensure_tensor_type(partial(_elwise, mode=Elemwise.Mode.POW))
eq = lambda x, y: ensure_tensor_type(_elwise)(x, y, mode=Elemwise.Mode.EQ).bool()  # noqa
ne = lambda x, y: logical_not(eq(x, y))  # noqa
le = lambda x, y: ensure_tensor_type(_elwise)(x, y, mode=Elemwise.Mode.LEQ).bool()  # noqa
lt = lambda x, y: ensure_tensor_type(_elwise)(x, y, mode=Elemwise.Mode.LT).bool()  # noqa
ge = lambda x, y: ensure_tensor_type(_elwise)(y, x, mode=Elemwise.Mode.LEQ).bool()  # swap x, y here # noqa
gt = lambda x, y: ensure_tensor_type(_elwise)(y, x, mode=Elemwise.Mode.LT).bool()  # swap x, y here # noqa


@ensure_tensor_type
def bitwise_left_shift(x, y):
    if not isinstance(x.dtype, (np.int32, np.uint8)):
        raise ValueError("Only Int Type tensor support left shift")
    return _elwise(x, y, mode=Elemwise.Mode.SHL)


@ensure_tensor_type
def bitwise_right_shift(x, y):
    if not isinstance(x.dtype, (np.int32, np.uint8)):
        raise ValueError("Only Int Type tensor support right shift")
    return _elwise(x, y, mode=Elemwise.Mode.SHR)


@ensure_tensor_type
def logical_and(x, y):
    return _elwise(x.astype("bool"), y.astype("bool"), mode=Elemwise.Mode.AND)


@ensure_tensor_type
def logical_or(x, y):
    return _elwise(x.astype("bool"), y.astype("bool"), mode=Elemwise.Mode.OR)


@ensure_tensor_type
def logical_xor(x, y):
    return _elwise(x.astype("bool"), y.astype("bool"), mode=Elemwise.Mode.XOR)


@ensure_tensor_type
def logical_not(x):
    return _elwise(x.astype("bool"), mode=Elemwise.Mode.NOT)
