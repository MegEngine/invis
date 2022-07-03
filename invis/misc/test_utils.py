#!/usr/bin/env python3

import numpy as np

import megengine as mge
import torch

import invis


def numpy_equal(mge_tensor, torch_tensor, rtol=1e-5):
    return np.allclose(
        mge_tensor.flatten().numpy(),
        torch_tensor.detach().flatten().cpu().numpy(),
        rtol=rtol,
    )


def iter_numpy_equal(mge_list, torch_list, rtol=1e-5):
    for mge_tensor, torch_tensor in zip(mge_list, torch_list):
        if not numpy_equal(mge_tensor, torch_tensor, rtol):
            return False
    return True


class TestModule:

    class Inner:

        def setUp(self):
            raise NotImplementedError

        def check_states(self):
            for k in self.states:
                self.assertTrue(
                    np.allclose(
                        getattr(self.module, k).flatten().numpy(),
                        getattr(self.torch_module, k).detach().flatten().cpu().numpy(),
                    )
                )

        def test_load(self):
            self.module.load_state_dict(self.weights)
            self.torch_module.load_state_dict(
                {k: torch.Tensor(v) for k, v in self.weights.items()}
            )
            self.check_states()

        def test_dump_load(self):
            self.module.load_state_dict(self.weights)
            weights = self.module.state_dict()

            self.torch_module.load_state_dict(
                {k: torch.Tensor(v) for k, v in weights.items()}
            )
            self.check_states()

        def test_mge_feature(self):
            # I don't like such mge feature
            x = mge.Tensor(1)
            y = mge.module.Linear(1, 1)
            self.module.temp = []
            prev_keys = set(self.module.state_dict().keys())
            self.module.temp.append(x)
            self.assertEqual(prev_keys, set(self.module.state_dict().keys()))
            self.module.temp.append(y)
            self.assertEqual(prev_keys, set(self.module.state_dict().keys()))

        def test_register_buffer(self):
            self.module.register_buffer("x", mge.Tensor(1))
            self.assertTrue("x" in self.module.state_dict().keys())

        def test_register_param(self):
            self.module.register_buffer("x", mge.Parameter(1))
            self.assertTrue("x" in self.module.state_dict().keys())

        def test_forward(self):
            module_input = getattr(self, "module_input", None)

            if module_input is not None:
                self.module.load_state_dict(self.weights)
                self.torch_module.load_state_dict(
                    {k: torch.Tensor(v) for k, v in self.weights.items()}
                )
                mge_out = self.module(invis.Tensor(module_input))
                torch_out = self.torch_module(torch.Tensor(module_input))
                if not isinstance(mge_out, (list, tuple, dict)):
                    self.assertIsInstance(mge_out, invis.Tensor)
                    self.assertTrue(numpy_equal(mge_out, torch_out))
                else:
                    if isinstance(mge_out, dict):
                        keys = mge_out.keys()
                        mge_out = [mge_out[k] for k in keys]
                        torch_out = [torch_out[k] for k in keys]
                    for mge_out_, torch_out_ in zip(mge_out, torch_out):
                        self.assertIsInstance(mge_out_, invis.Tensor)
                        self.assertTrue(numpy_equal(mge_out_, torch_out_))
