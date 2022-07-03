#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

from invis import nn
from invis.misc import TestModule


class TestPixelUnshuffle(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        shuffle_dim = 3
        self.module = nn.PixelUnshuffle(shuffle_dim)
        self.torch_module = torchnn.PixelUnshuffle(shuffle_dim)
        self.weights = {}
        self.states = list(self.weights.keys())
        self.module_input = np.random.normal(
            size=(2, 4, 1, 2 * shuffle_dim, 3 * shuffle_dim)
        ).astype("float32")


if __name__ == "__main__":
    unittest.main()
