#!/usr/bin/env python3

from megengine.device import is_cuda_available

__all__ = [
    "is_available",
]


def is_available():
    return is_cuda_available()
