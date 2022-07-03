#!/usr/bin/env python3

import unittest

import numpy as np

import torch

import invis
from invis.misc import TestModule


class TestPool(TestModule.Inner, unittest.TestCase):

    def setUp(self):
        self.module = invis.nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.torch_module = torch.nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.module_input = np.random.rand(1, 1, 112, 112)
        self.weights = {}
        self.states = list(self.weights.keys())


if __name__ == "__main__":
    unittest.main()
