#!/usr/bin/env python3

import unittest

import numpy as np

import torch.nn as torchnn

import invis
import invis.nn as nn


class TestContainer(unittest.TestCase):

    def setUp(self):
        self.conv = nn.Conv2d(3, 4, 1)
        self.norm = nn.BatchNorm2d(4)
        self.inputs = invis.rand(2, 3, 3, 3)
        self.seq = nn.Sequential(self.conv, self.norm)
        self.seq.eval()

    def test_seq(self):
        m1 = nn.Sequential(self.conv)
        m1.add_module("norm", self.norm)
        out1 = m1(self.inputs)
        out2 = self.seq(self.inputs)
        self.assertIsInstance(out2, invis.Tensor)
        self.assertTrue(
            np.allclose(out1.numpy(), out2.numpy())
        )
        m1.eval()
        self.assertFalse(m1[0].training)
        self.assertFalse(m1[1].training)

    def test_module_dict(self):
        conv = self.conv
        norm = self.norm

        class MgeModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                    "conv": conv,
                    "conv1": nn.Conv2d(3, 4, 1),
                })
                self.norm = nn.ModuleDict([
                    ["norm", norm],
                    ["norm1", nn.BatchNorm2d(4)]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.norm[act](x)
                return x

        class TorchModule(torchnn.Module):
            def __init__(self):
                super().__init__()
                self.choices = torchnn.ModuleDict({
                    "conv": torchnn.Conv2d(3, 4, 1),
                    "conv1": torchnn.Conv2d(3, 4, 1),
                })
                self.norm = torchnn.ModuleDict([
                    ["norm", torchnn.BatchNorm2d(4)],
                    ["norm1", torchnn.BatchNorm2d(4)]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.norm[act](x)
                return x

        m = MgeModule()
        out1 = m(self.inputs, "conv", "norm")
        out2 = self.seq(self.inputs)
        self.assertTrue(
            np.allclose(out1.numpy(), out2.numpy())
        )

        torch_m = TorchModule()
        weights = {
            k: np.random.rand(*v.shape) for k, v in torch_m.state_dict().items()
            if "num_batches_tracked" not in k
        }
        import megengine as mge
        m.load_state_dict(
            {k: mge.Tensor(v) for k, v in weights.items()},
        )
        from torch import Tensor
        torch_m.load_state_dict({k: Tensor(v) for k, v in weights.items()})

        torch_m.eval()
        m.eval()
        torch_out = torch_m(Tensor(self.inputs.numpy()), "conv", "norm")
        mge_out = m(self.inputs, "conv", "norm")
        self.assertIsInstance(mge_out, invis.Tensor)
        self.assertTrue(np.allclose(
            torch_out.detach().cpu().numpy(), mge_out.numpy()
        ))

    def test_module_list(self):
        m_list = nn.ModuleList([self.conv])
        m_list.append(self.norm)
        m_list.eval()
        out1 = self.inputs
        for m in m_list:
            out1 = m(out1)

        out2 = self.seq(self.inputs)
        self.assertIsInstance(out1, invis.Tensor)
        self.assertTrue(
            np.allclose(out1.numpy(), out2.numpy())
        )
        self.assertIsInstance(m_list[0], nn.Conv2d)


if __name__ == "__main__":
    unittest.main()
