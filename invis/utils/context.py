#!/usr/bin/env python3

import contextlib

__all__ = [
    "no_grad",
]


@contextlib.contextmanager
def nullcontext():
    yield


no_grad = nullcontext
