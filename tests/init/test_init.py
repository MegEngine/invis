#!/usr/bin/env python3

import unittest

import numpy as np

import torch
import torch.nn.init as torch_init

import invis
from invis.nn import init


class TesetConv(unittest.TestCase):

    def setUp(self):
        self.mge_x = invis.randn(3, 16, 10, 10)
        self.torch_x = torch.randn(3, 16, 10, 10)

    def test_normal(self):
        init.normal_(self.mge_x)

    def test_uniform(self):
        init.uniform_(self.mge_x)

    def test_kaiming(self):
        init.kaiming_normal_(self.mge_x)
        init.kaiming_uniform_(self.mge_x)

    def test_xavier(self):
        init.xavier_normal_(self.mge_x)
        init.xavier_uniform_(self.mge_x)

    def test_dirac(self):
        init.dirac_(self.mge_x)
        torch_init.dirac_(self.torch_x)

        self.assertTrue(
            np.allclose(self.mge_x.numpy(), self.torch_x.detach().cpu().numpy())
        )

    def test_constant_(self):
        init.constant_(self.mge_x, 3.4)
        torch_init.constant_(self.torch_x, 3.4)

        self.assertTrue(
            np.allclose(self.mge_x.numpy(), self.torch_x.detach().cpu().numpy())
        )

    def test_zeros(self):
        init.zeros_(self.mge_x)
        torch_init.zeros_(self.torch_x)

        self.assertTrue(
            np.allclose(self.mge_x.numpy(), self.torch_x.detach().cpu().numpy())
        )

    def test_ones(self):
        init.ones_(self.mge_x)
        torch_init.ones_(self.torch_x)

        self.assertTrue(
            np.allclose(
                self.mge_x.numpy(), self.torch_x.detach().cpu().numpy(),
            )
        )

    def test_eyes(self):
        mge_x = invis.rand(10, 10)
        torch_x = torch.rand(10, 10)
        init.eye_(mge_x)
        torch_init.eye_(torch_x)

        self.assertTrue(
            np.allclose(mge_x.numpy(), torch_x.detach().cpu().numpy())
        )


if __name__ == "__main__":
    unittest.main()
