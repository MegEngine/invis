#!/usr/bin/env python3

import unittest

import numpy as np

import torch

import invis as inv
from invis.misc.test_utils import iter_numpy_equal, numpy_equal


class TestRedefFunc(unittest.TestCase):

    def setUp(self):
        self.array = np.random.random(size=(2, 3, 4, 5)).astype("float32")
        self.mge_tensor = inv.from_numpy(self.array)
        self.torch_tensor = torch.from_numpy(self.array)
        np.random.seed(3)

    def check_value_and_type(self, mge_out, torch_out, rtol=1e-5, func_name=None):
        prefix = "output" if func_name is None else f"output of {func_name}"

        type_msg = prefix + " is not invis Tensor"
        value_msg = prefix + " is not matched with torch"

        if isinstance(mge_out, (tuple, list)):
            self.assertTrue(iter_numpy_equal(mge_out, torch_out), msg=value_msg)
            for v in mge_out:
                self.assertIsInstance(v, inv.Tensor, msg=type_msg)
        else:
            self.assertIsInstance(mge_out, inv.Tensor, msg=type_msg)
            self.assertTrue(numpy_equal(mge_out, torch_out, rtol=rtol), msg=value_msg)

    def diff_check(self, func_name, *args, torch_namespace=torch, rtol=1e-5, **kwargs):
        torch_f = getattr(torch_namespace, func_name)
        torch_out = torch_f(self.torch_tensor, *args, **kwargs)

        mge_f = getattr(inv, func_name)
        mge_out = mge_f(self.mge_tensor, *args, **kwargs)
        self.check_value_and_type(mge_out, torch_out, rtol, func_name)

    def diff_check_no_tensor(self, func_name, *args, torch_namespace=torch, rtol=1e-5, **kwargs):
        torch_f = getattr(torch_namespace, func_name)
        torch_out = torch_f(*args, **kwargs)

        mge_f = getattr(inv, func_name)
        mge_out = mge_f(*args, **kwargs)
        self.check_value_and_type(mge_out, torch_out, rtol, func_name)

    def test_permute(self):
        mge_out = inv.permute(self.mge_tensor, (1, 3, 2, 0))
        self.assertIsInstance(mge_out, inv.Tensor)
        torch_out = self.torch_tensor.permute(1, 3, 2, 0)
        self.assertTrue(numpy_equal(mge_out, torch_out))

    def test_transpose(self):
        self.diff_check("transpose", 1, 3)
        self.diff_check("transpose", 0, -1)
        self.diff_check("transpose", 3, -1)

    def test_chunk(self):
        mge_out = [x.numpy() for x in inv.chunk(self.mge_tensor, 2, dim=2)]
        torch_out = [x.cpu().numpy() for x in torch.chunk(self.torch_tensor, 2, dim=2)]
        for mge_out_, torch_out_ in zip(mge_out, torch_out):
            assert np.allclose(torch_out_, mge_out_)

        mge_out = [x.numpy() for x in inv.chunk(self.mge_tensor, 2, dim=3)]
        torch_out = [x.cpu().numpy() for x in torch.chunk(self.torch_tensor, 2, dim=3)]
        for mge_out_, torch_out_ in zip(mge_out, torch_out):
            assert np.allclose(torch_out_, mge_out_)

    def test_split(self):
        mge_out = inv.split(self.mge_tensor, 1)
        torch_out = torch.split(self.torch_tensor, 1)

        mge_out = inv.split(self.mge_tensor, [2, 3], dim=-1)
        torch_out = torch.split(self.torch_tensor, [2, 3], dim=-1)
        self.check_value_and_type(mge_out, torch_out, func_name="split")

    def test_interpolate(self):
        self.diff_check("interpolate", torch_namespace=torch.nn.functional, scale_factor=2)

    def test_pad(self):
        pad_shape = [(1, 2), (1, 0, 2, 1), (1, 2, 2, 1, 0, 1), (-1, -2), (-2, 1, -2, -1)]
        pad_mode = ["constant", "reflect", "replicate"]
        for pad in pad_shape:
            for mode in pad_mode:
                if mode != "constant" and len(pad) == 2:
                    continue
                self.diff_check("pad", pad, torch_namespace=torch.nn.functional, mode=mode)

    def test_mean(self):
        self.diff_check("mean")
        self.diff_check("mean", dim=(2, 3))
        self.diff_check("mean", dim=(2, 3), keepdim=True)

    def test_var(self):
        self.diff_check("var")
        self.diff_check("var", dim=2)
        self.diff_check("var", dim=3, unbiased=False)
        self.diff_check("var", dim=3, unbiased=False, keepdim=True)

    def test_std(self):
        self.diff_check("std")
        self.diff_check("std", dim=2)
        self.diff_check("std", dim=3, unbiased=False)
        self.diff_check("std", dim=3, unbiased=False, keepdim=True)

    def test_round(self):
        self.diff_check("round")
        if "decimals" in torch.round.__doc__:  # some version of torch doesn't support decimals
            self.diff_check("round", decimals=2)
            self.diff_check("round", decimals=3)

    def test_flatten(self):
        self.diff_check("flatten")
        self.diff_check("flatten", start_dim=1)
        self.diff_check("flatten", end_dim=-2)

    def test_linspace(self):
        self.diff_check_no_tensor("linspace", 0, 10, 5)
        self.diff_check_no_tensor("linspace", -10, 10, steps=5)
        self.diff_check_no_tensor("linspace", 3, 10, steps=1)

    def test_logspace(self):
        self.diff_check_no_tensor("logspace", start=-10, end=10, steps=5)
        self.diff_check_no_tensor("logspace", 0.1, 1.0, steps=5)
        self.diff_check_no_tensor("logspace", 0.1, 1.0, steps=5, base=2)

    def test_arange(self):
        self.diff_check_no_tensor("arange", 5)
        self.diff_check_no_tensor("arange", 1, 4)
        self.diff_check_no_tensor("arange", 1, 2.5, 0.5)

    def test_topk(self):
        self.mge_tensor = self.mge_tensor.reshape(4, -1, 2)
        self.torch_tensor = self.torch_tensor.reshape(4, -1, 2)
        self.diff_check("topk", k=5, dim=1)

    def test_sort(self):
        self.mge_tensor = self.mge_tensor.reshape(4, -1, 2, 3)
        self.torch_tensor = self.torch_tensor.reshape(4, -1, 2, 3)
        self.diff_check("sort", dim=1)

    def test_range(self):
        self.diff_check_no_tensor("range", 0, 10)
        self.diff_check_no_tensor("range", 0, 5, 0.5)
        self.diff_check_no_tensor("range", 0, 10.1, 0.5)

    def test_meshgrid(self):
        hsize, wsize = 10, 20
        mge_out = inv.meshgrid([inv.arange(hsize), inv.arange(wsize)], indexing="ij")
        torch_out = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
        self.check_value_and_type(mge_out, torch_out, func_name="meshgrid")

    def test_prod(self):
        self.diff_check("prod")
        self.diff_check("prod", dim=2)
        self.diff_check("prod", dim=2, keepdim=True)

    def test_sum(self):
        self.diff_check("sum")
        self.diff_check("sum", dim=2)
        self.diff_check("sum", dim=2, keepdim=True)

    def test_cumsum(self):
        self.diff_check("cumsum", dim=2)

    def test_cumprod(self):
        self.diff_check("cumprod", dim=2)

    def test_avg_pool2d(self):
        kwargs = {"torch_namespace": torch.nn.functional}
        self.diff_check("avg_pool2d", 3, 2, 1, ceil_mode=False, **kwargs)
        self.diff_check("avg_pool2d", 2, 2, 0, count_include_pad=False, **kwargs)

    def test_max_pool2d(self):
        kwargs = {"torch_namespace": torch.nn.functional}
        self.diff_check("max_pool2d", 3, 2, 1, ceil_mode=False, **kwargs)
        self.diff_check("max_pool2d", 3, 2, 1, ceil_mode=True, **kwargs)

    def test_isnan(self):
        torch_out = torch.isnan(torch.tensor([1, float('nan'), float("inf")]))
        mge_out = inv.isnan(inv.Tensor([1, float("nan"), float("inf")]))
        self.check_value_and_type(mge_out, torch_out, func_name="isnan")

    def test_isinf(self):
        torch_out = torch.isinf(torch.tensor([1, float('nan'), float("inf")]))
        mge_out = inv.isinf(inv.Tensor([1, float("nan"), float("inf")]))
        self.check_value_and_type(mge_out, torch_out, func_name="isinf")

    def test_isfinite(self):
        torch_out = torch.isfinite(torch.tensor([1, float('nan'), float("inf")]))
        mge_out = inv.isfinite(inv.Tensor([1, float("nan"), float("inf")]))
        self.check_value_and_type(mge_out, torch_out, func_name="isfinite")

    def test_min(self):
        self.diff_check("min")
        self.diff_check("min", dim=-2)
        self.diff_check("min", dim=3, keepdim=True)

    def test_max(self):
        self.diff_check("max")
        self.diff_check("max", dim=-2)
        self.diff_check("max", dim=3, keepdim=True)

    def test_minimum(self):
        np_value = np.random.normal(size=(2, 3, 4, 5))
        torch_out = torch.minimum(self.torch_tensor, torch.Tensor(np_value))
        mge_out = inv.minimum(self.mge_tensor, inv.Tensor(np_value))
        self.check_value_and_type(mge_out, torch_out, func_name="minimum")

    def test_maximum(self):
        np_value = np.random.normal(size=(2, 3, 4, 5))
        torch_out = torch.maximum(self.torch_tensor, torch.Tensor(np_value))
        mge_out = inv.maximum(self.mge_tensor, inv.Tensor(np_value))
        self.check_value_and_type(mge_out, torch_out, func_name="maximum")

    def test_nonzero(self):
        import megengine.functional as F

        np_value = np.random.randint(low=0, high=3, size=(2, 3, 4, 5))
        torch_input = torch.Tensor(np_value)
        mge_input = inv.Tensor(np_value)

        torch_out = torch.nonzero(torch_input)
        mge_out = inv.nonzero(mge_input)
        self.check_value_and_type(mge_out, torch_out, func_name="nonzero")

        torch_out = torch.nonzero(torch_input, as_tuple=True)
        mge_out = inv.nonzero(mge_input, as_tuple=True)
        self.check_value_and_type(mge_out, torch_out, func_name="nonzero")

        # make sure as_tuple also works with __getitem__
        value = F.cond_take(mge_input != 0, mge_input)[0]
        self.assertTrue(inv.equal(mge_input[mge_out], value))

    def test_where(self):
        np_val = np.random.random(size=(2, 3, 4, 5)).astype("float32")
        torch_input = torch.Tensor(np_val)
        mge_input = inv.Tensor(np_val)

        torch_out = torch.where(torch_input > 0.5)
        mge_out = inv.where(mge_input > 0.5)
        self.check_value_and_type(mge_out, torch_out, func_name="where")

        torch_out = torch.where(torch_input > 0.5, torch_input, self.torch_tensor)
        mge_out = inv.where(mge_input > 0.5, mge_input, self.mge_tensor)
        self.check_value_and_type(mge_out, torch_out, func_name="where")

    def test_tile(self):
        self.diff_check("tile", (1, 2, 3, 4))
        self.diff_check("tile", (3, 2))
        self.diff_check("tile", (2,))
        self.diff_check("tile", (2, 2, 3, 4, 5))

    def test_repeat_interleave(self):
        self.diff_check("repeat_interleave", 2, dim=-1)

        torch_out = torch.repeat_interleave(self.torch_tensor, torch.tensor([1, 2, 3]), dim=1)
        mge_out = inv.repeat_interleave(self.mge_tensor, inv.tensor([1, 2, 3]), dim=1)
        self.check_value_and_type(mge_out, torch_out, func_name="repeat_interleave")

    def test_broadcast_to(self):
        self.mge_tensor = self.mge_tensor.reshape(4, 1, -1, 2)
        self.torch_tensor = self.torch_tensor.reshape(4, 1, -1, 2)

        self.diff_check("broadcast_to", (4, 1, 15, 2))
        self.diff_check("broadcast_to", (2, -1, -1, -1, 2))
        self.diff_check("broadcast_to", (2, 3, 4, 2, -1, 2))

    def test_gather(self):
        index = np.random.randint(low=0, high=3, size=(2, 3, 4, 5))
        torch_out = torch.gather(self.torch_tensor, dim=1, index=torch.tensor(index))
        mge_out = inv.gather(self.mge_tensor, dim=1, index=inv.tensor(index))
        self.check_value_and_type(mge_out, torch_out, func_name="gather")

    def test_scatter(self):
        index = np.random.randint(low=0, high=3, size=(2, 3, 4, 5))
        source = np.random.normal(size=(2, 3, 4, 5)).astype("float32")
        torch_out = torch.scatter(
            self.torch_tensor, 1, torch.tensor(index), torch.tensor(source)
        )
        mge_out = inv.scatter(
            self.mge_tensor, 1, inv.tensor(index), inv.tensor(source)
        )
        self.check_value_and_type(mge_out, torch_out, func_name="scatter")


if __name__ == "__main__":
    unittest.main()
