#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

from invis import nn
from invis.misc import TestModule


class TestPixelShuffle(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        shuffle_dim = 3
        self.module = nn.PixelShuffle(shuffle_dim)
        self.torch_module = torchnn.PixelShuffle(shuffle_dim)
        self.weights = {}
        self.states = list(self.weights.keys())
        self.module_input = np.random.normal(
            size=(2, shuffle_dim * shuffle_dim, 2, 3)
        ).astype("float32")


if __name__ == "__main__":
    unittest.main()
