#!/usr/bin/env python3


import unittest

import numpy as np

import torch

import invis as inv


class TestRedefFunc(unittest.TestCase):

    def setUp(self):
        self.array = np.random.random(size=(2, 3, 4, 5))
        self.mge_tensor = inv.from_numpy(self.array)
        self.torch_tensor = torch.from_numpy(self.array)

    def check_diff(self, func_name, *args, **kwargs):
        mge_f = getattr(self.mge_tensor, func_name)
        mge_out = mge_f(*args, **kwargs)

        torch_f = getattr(self.torch_tensor, func_name)
        torch_out = torch_f(*args, **kwargs)
        self.assertTrue(
            np.allclose(torch_out, mge_out.numpy()),
            msg=f"output of {func_name} is not matched with torch"
        )

    def test_max(self):
        self.check_diff("max")

    def test_mean(self):
        self.check_diff("mean")

    def test_var(self):
        self.check_diff("var")

    def test_std(self):
        self.check_diff("std")


if __name__ == "__main__":
    unittest.main()
