#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

from invis import nn
from invis.misc import TestModule


class TestConv(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        self.module = nn.ConvTranspose2d(3, 3, 3, bias=True, groups=3)
        self.torch_module = torchnn.ConvTranspose2d(3, 3, 3, bias=True, groups=3)
        self.weights = {
            "weight": np.random.normal(size=(3, 1, 3, 3)).astype("float32"),
            "bias": np.random.normal(size=3).astype("float32"),
        }
        self.states = list(self.weights.keys())
        self.module_input = np.random.normal(size=(1, 3, 3, 3)).astype("float32")


if __name__ == "__main__":
    unittest.main()
