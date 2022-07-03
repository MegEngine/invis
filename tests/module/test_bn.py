#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

from invis import nn
from invis.misc import TestModule


class TestBN(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        norm_dim = 10
        self.module = nn.BatchNorm2d(norm_dim)
        self.torch_module = torchnn.BatchNorm2d(norm_dim)
        self.weights = {
            "weight": np.random.normal(size=(norm_dim)).astype("float32"),
            "bias": np.random.normal(size=(norm_dim)).astype("float32"),
            "running_mean": np.random.normal(size=(norm_dim)).astype("float32"),
            "running_var": np.random.normal(size=(norm_dim)).astype("float32"),
        }
        self.states = ["weight", "bias", "running_mean", "running_var"]
        self.module_input = np.random.normal(size=(2, norm_dim, 4, 4)).astype("float32")
        np.random.seed(3)


if __name__ == "__main__":
    unittest.main()
