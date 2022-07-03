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
        np.random.seed(3)

    def inner(self, func_name, other):
        inv_func = getattr(inv, func_name)
        torch_func = getattr(torch, func_name)
        mge_out = inv_func(self.mge_tensor, inv.Tensor(other))
        self.assertIsInstance(mge_out, inv.Tensor)

        torch_out = torch_func(self.torch_tensor, torch.Tensor(other)).detach().cpu().numpy()
        self.assertTrue(
            np.allclose(mge_out.numpy(), torch_out), msg=f"{func_name} output not matched"
        )

    def test_binary_elemwise(self):

        func_names = [
            "add", "sub", "mul", "true_divide", "floor_divide", "pow",
        ]
        others = [[0.1], np.random.random(size=(2, 3, 4, 5))]
        for name in func_names:
            for other in others:
                self.inner(name, other)

    def test_cmp_func(self):
        func_names = ["lt", "gt", "le", "ge", "eq", "ne"]
        others = [[0.5], np.random.random(size=(2, 3, 4, 5))]
        for name in func_names:
            for other in others:
                self.inner(name, other)

    def test_logical_elemwise(self):
        def inner(func_name, other):
            inv_func = getattr(inv, func_name)
            torch_func = getattr(torch, func_name)
            args = () if "not" in func_name else (inv.Tensor(other),)
            mge_out = inv_func(self.mge_tensor, *args)
            self.assertIsInstance(mge_out, inv.Tensor)

            args = () if "not" in func_name else (torch.Tensor(other),)
            torch_out = torch_func(self.torch_tensor, *args).detach().cpu().numpy()
            self.assertTrue(np.allclose(mge_out.numpy(), torch_out))

        logical = ["not", "and", "or", "xor"]
        func_names = ["logical_" + x for x in logical]
        shape = (2, 3, 4, 5)
        others = [[0], np.ones(shape), np.zeros(shape), np.random.random(size=shape)]
        for name in func_names:
            for other in others:
                inner(name, other)


if __name__ == "__main__":
    unittest.main()
