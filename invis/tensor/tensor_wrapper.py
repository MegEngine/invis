#!/usr/bin/env python3

import functools

import numpy as np

import megengine as mge

import invis

from . import function as F
from .function.helper import ensure_tensor_type, inplace, swap_argument

__all__ = [
    "Tensor",
    "tensor",
]


class Tensor(mge.Tensor):

    def __new__(cls, *args, **kwargs):
        # TODO: add modify wrapper
        if "device" in kwargs:
            device = kwargs["device"]
            kwargs["device"] = invis.device(device)
        elif len(args) > 2:
            args = list(args)
            args[2] = invis.device(args[2])
            args = tuple(args)

        return mge.Tensor.__new__(cls, *args, **kwargs)

    def dim(self):
        return self.ndim

    def numel(self):
        r"""Returns the size of the self :class:`~.Tensor`.
        The returned value is a subclass of :class:`tuple`.
        """
        shape = self.shape
        if shape.__class__ is tuple:
            return np.prod(self.shape).item()
        return shape.prod()

    def type(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
            if dtype == np.float32:
                return "FloatTensor"
            elif dtype == np.int32:
                return "IntTensor"
            elif dtype == np.bool_:
                return "BoolTensor"
            elif dtype == np.float16:
                return "HalfTensor"
            elif dtype == np.uint8:
                return "ByteTensor"
            else:
                return dtype
        else:
            if dtype == "FloatTensor":
                return self.float()
            elif dtype == "IntTensor":
                return self.int()
            elif dtype == "BoolTensor":
                return self.bool()
            elif dtype == "HalfTensor":
                return self.half()
            elif dtype == "ByteTensor":
                return self.byte()
            else:
                return self

    def size(self, dim=None):
        s = self.shape
        return s[dim] if dim is not None else s

    def reshape(self, *shape):
        return F.reshape(self, shape)

    def reshape_as(self, other):
        return F.reshape(self, other.shape)

    def view(self, *shape):
        # megengine doesn't support view in lower level
        return F.reshape(self, shape)

    def view_as(self, other):
        # megengine doesn't support view in lower level
        return self.reshape_as(other)

    def cpu(self):
        return self.to("cpu1")

    def cuda(self):
        return self

    @ensure_tensor_type
    def float(self):
        return self.astype("float32")

    @ensure_tensor_type
    def half(self):
        return self.astype("float16")

    @ensure_tensor_type
    def int(self):
        return self.astype("int32")

    @ensure_tensor_type
    def byte(self):
        return self.astype("uint8")

    @ensure_tensor_type
    def bool(self):
        return self.astype("bool")

    @ensure_tensor_type
    def to(self, *args, **kwargs):
        first_arg = args[0]
        if "cuda" in first_arg or "cpu" in first_arg:
            list_args = list(args)
            list_args[0] = invis.device(first_arg)
            return mge.Tensor.to(self, *tuple(list_args), **kwargs)
        else:  # dtype
            return self.astype(*args, **kwargs)

    def argmax(self, dim=None, keepdim=False):
        return F.argmax(self, dim, keepdim)

    def eq(self, other):
        return F.eq(self, other)

    def squeeze(self, dim=None):
        return F.squeeze(self, dim)

    def unsqueeze(self, dim):
        return F.unsqueeze(self, dim)

    def contiguous(self):
        # megengine doesn't support view in lower level
        return self

    def chunk(self, chunks, dim=0):
        return F.chunk(self, chunks, dim)

    @ensure_tensor_type
    def transpose(self, dim0, dim1):
        # swap axis
        # TODO: this should be solved by mge
        if dim0 == dim1:
            return self

        ndim = self.ndim
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim
        pattern = list(range(ndim))
        pattern[dim0] = dim1
        pattern[dim1] = dim0
        return mge.Tensor.transpose(self, pattern)

    def tile(self, *reps):
        return F.tile(self, reps)

    def repeat(self, *sizes):
        if len(sizes) < self.ndim:
            raise RuntimeError(
                "Number of dimensions of repeat dims can not be \
                 smaller than number of dimensions of tensor"
            )
        else:
            return F.tile(self, sizes)

    def permute(self, *dims):
        return F.permute(self, dims)

    def expand(self, *sizes):
        return F.broadcast_to(self, sizes)

    def __repr__(self):
        # NOTE: I don't like repr in megengine
        piece = "{}(".format(self.__class__.__name__)
        leading_space = " " * len(piece)
        with np.printoptions(precision=4, suppress=True):
            numpy_string = str(self.numpy()).replace("\n", "\n" + leading_space)
            piece += numpy_string
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        piece += ", device={}".format(self.device.physical_name) + ")"
        return piece

    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except Exception as e:
            # TODO It's tricky here
            inplace_func = False
            if name.endswith("_"):
                inplace_func = True
                name = name[:-1]
            func = getattr(F, name, None)
            if func is not None and inplace_func:
                func = inplace(func)

            if func is not None:
                return functools.partial(F.ensure_tensor_type(func), self)
            else:
                raise e

    # some other functions
    relu = F.relu
    sigmoid = F.sigmoid
    t = F.t
    roll = F.roll

    # functions for convience
    flatten = F.flatten
    all = F.all
    sum = F.sum
    mean = F.mean
    std = F.std
    var = F.var
    round = F.round
    min = F.min
    max = F.max

    clamp = F.clamp
    clamp_ = F.clamp_
    masked_fill = F.masked_fill
    masked_fill_ = F.masked_fill_

    __getitem__ = ensure_tensor_type(mge.Tensor.__getitem__)
    # innernal function
    __lt__ = F.lt
    __le__ = F.le
    __gt__ = F.gt
    __ge__ = F.ge
    __eq__ = F.eq
    __ne__ = F.ne

    __add__ = F.add
    __sub__ = F.sub
    __mul__ = F.mul
    # TODO: calling matmul might harm the performance
    __matmul__ = F.matmul
    __truediv__ = F.true_divide
    __floordiv__ = F.floor_divide
    __mod__ = F.fmod
    # __divmode__
    __pow__ = F.pow
    __lshift__ = F.bitwise_left_shift
    __rshift__ = F.bitwise_right_shift
    __and__ = F.logical_and
    __or__ = F.logical_or
    __xor__ = F.logical_xor

    __radd__ = swap_argument(__add__)
    __rsub__ = swap_argument(__sub__)
    __rmul__ = swap_argument(__mul__)
    __rmatmul__ = swap_argument(__matmul__)
    __rtruediv__ = swap_argument(__truediv__)
    __rfloordiv__ = swap_argument(__floordiv__)
    __rmod__ = swap_argument(__mod__)
    __rpow__ = swap_argument(__pow__)
    __rlshift__ = swap_argument(__lshift__)
    __rrshift__ = swap_argument(__rshift__)
    __rand__ = swap_argument(__and__)
    __ror__ = swap_argument(__or__)
    __rxor__ = swap_argument(__xor__)

    __iadd__ = inplace(__add__)
    __isub__ = inplace(__sub__)
    __imul__ = inplace(__mul__)
    __imatmul__ = inplace(__matmul__)
    __itruediv__ = inplace(__truediv__)
    __ifloordiv__ = inplace(__floordiv__)
    __imod__ = inplace(__mod__)
    __ipow__ = inplace(__pow__)
    __ilshift__ = inplace(__lshift__)
    __irshift__ = inplace(__rshift__)
    __iand__ = inplace(__and__)
    __ior__ = inplace(__or__)
    __ixor__ = inplace(__xor__)


tensor = Tensor
