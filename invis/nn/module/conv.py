#!/usr/bin/env python3

from collections import OrderedDict

import megengine as mge
import megengine.module as M
from megengine.module.module import _is_buffer

import invis
from invis.utils.wrapper import is_torch_tensor

from .patch import patch_attribute, patch_method

__all__ = ["Conv2d", "ConvTranspose2d"]


def _is_parameter(obj):
    return isinstance(obj, (invis.nn.Parameter, mge.Parameter))


class Conv2d(M.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # users have already override `_state_dict` method, so here code
        # using `M.Conv2d._state_dict` but not `self._state_dict`
        local_state_dict = M.Conv2d._state_dict(self, keep_var=True)
        weights_to_load = {k: state_dict[prefix + k] for _, k in local_state_dict}
        for k, v in weights_to_load.items():
            if is_torch_tensor(v):
                weights_to_load[k] = v.detach().cpu().numpy()

        for (module_type, k), var in local_state_dict.items():
            if k not in weights_to_load:
                missing_keys.append(k)
                continue

            to_be_load = weights_to_load[k]
            if var.shape != to_be_load.shape:
                if "bias" in k or self.groups > 1:
                    to_be_load = to_be_load.reshape(var.shape)
                else:
                    raise ValueError(
                        f"param `{k}` shape mismatch, should be {var.shape}, get {to_be_load.shape}"
                    )
            var._reset(
                type(var)(
                    to_be_load, dtype=to_be_load.dtype, device=var.device, no_cache=True
                )
            )

    def _state_dict(self, rst=None, prefix="", keep_var=False):
        r"""Returns a dictionary containing whole states of the module."""

        def is_state(obj):
            return _is_parameter(obj) or _is_buffer(obj)

        module_type = self.__class__
        if rst is None:
            rst = OrderedDict()

        for k, v in self._flatten(recursive=False, with_key=True, predicate=is_state):
            assert prefix + k not in rst, "duplicated state: {}".format(k)
            if k == "bias":  # conv bias
                v = v.reshape(-1)  # torch bias is 1-dim
            elif k == "weight" and self.groups > 1:
                *_, out_c, kh, kw = v.shape
                v = v.reshape(-1, out_c, kh, kw)

            if keep_var:
                rst[(module_type, prefix + k)] = v
            else:
                rst[(module_type, prefix + k)] = v.numpy()

        for k, submodule in self._flatten(
            recursive=False,
            with_key=True,
            predicate=lambda obj: isinstance(obj, M.Module),
        ):
            submodule.state_dict(rst, prefix + k + ".", keep_var)

        return rst


Conv2d = patch_method(Conv2d, patch_override=False)


class ConvTranspose2d(M.ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)
        self._parameters["weight"] = self.weight
        if self.bias is not None:
            self._parameters["bias"] = self.bias

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # users have already override `_state_dict` method, so here code
        # using `M.ConvTranspose2d._state_dict` but not `self._state_dict`
        local_state_dict = M.ConvTranspose2d._state_dict(self, keep_var=True)
        weights_to_load = {k: state_dict[prefix + k] for _, k in local_state_dict}
        for k, v in weights_to_load.items():
            if is_torch_tensor(v):
                weights_to_load[k] = v.detach().cpu().numpy()

        for (module_type, k), var in local_state_dict.items():
            if k not in weights_to_load:
                missing_keys.append(k)
                continue

            to_be_load = weights_to_load[k]
            if var.shape != to_be_load.shape:
                if "bias" in k or self.groups > 1:
                    to_be_load = to_be_load.reshape(var.shape)
                else:
                    raise ValueError(
                        f"param `{k}` shape mismatch, should be {var.shape}, get {to_be_load.shape}"
                    )
            var._reset(
                type(var)(
                    to_be_load, dtype=to_be_load.dtype, device=var.device, no_cache=True
                )
            )

    def _state_dict(self, rst=None, prefix="", keep_var=False):
        r"""Returns a dictionary containing whole states of the module."""

        def is_state(obj):
            return _is_parameter(obj) or _is_buffer(obj)

        module_type = self.__class__
        if rst is None:
            rst = OrderedDict()

        for k, v in self._flatten(recursive=False, with_key=True, predicate=is_state):
            assert prefix + k not in rst, "duplicated state: {}".format(k)
            if k == "bias":  # conv bias
                v = v.reshape(-1)  # torch bias is 1-dim
            elif k == "weight" and self.groups > 1:
                *_, out_c, kh, kw = v.shape
                v = v.reshape(-1, out_c, kh, kw)

            if keep_var:
                rst[(module_type, prefix + k)] = v
            else:
                rst[(module_type, prefix + k)] = v.numpy()

        for k, submodule in self._flatten(
            recursive=False,
            with_key=True,
            predicate=lambda obj: isinstance(obj, M.Module),
        ):
            submodule.state_dict(rst, prefix + k + ".", keep_var)

        return rst


ConvTranspose2d = patch_method(ConvTranspose2d, patch_override=False)
