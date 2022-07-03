#!/usr/bin/env python3

import pickle

import megengine as mge
import torch
from megengine import save

__all__ = [
    "manual_seed",
    "device",
    "save",
    "load",
]


def manual_seed(seed):
    mge.random.seed(seed)


class device(str):

    def __repr__(self):
        return f"Device('{self}')"


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    weights = torch.load(f, map_location="cpu", **pickle_load_args)

    def convert2np(weights: dict):
        """convert tensor to numpy recursively"""
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                weights[k] = v.detach().numpy()
            elif isinstance(v, dict):
                weights[k] = convert2np(v)

        return weights

    return convert2np(weights)


def is_torch_tensor(obj):
    return isinstance(obj, torch.Tensor)
