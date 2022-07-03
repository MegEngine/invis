#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

from invis import nn
from invis.misc import TestModule


class TestLinear(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        self.module = nn.Linear(3, 4, bias=True)
        self.torch_module = torchnn.Linear(3, 4, bias=True)
        self.weights = {
            "weight": np.random.normal(size=(4, 3)).astype("float32"),
            "bias": np.random.normal(size=4).astype("float32"),
        }
        self.states = list(self.weights.keys())
        self.module_input = np.random.normal(size=(10, 3)).astype("float32")


if __name__ == "__main__":
    unittest.main()
