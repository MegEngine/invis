#!/usr/bin/env python3

import math

import megengine.module as M
from megengine.module import Linear as MGE_Linear

from invis.utils.wrapper import is_torch_tensor

from .patch import patch_attribute, patch_method

__all__ = ["Linear"]


class Linear(MGE_Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)
        self._parameters["weight"] = self.weight
        if self.bias is not None:
            self._parameters["bias"] = self.bias

    def reset_parameters(self) -> None:
        M.init.msra_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = M.init.calculate_fan_in_and_fan_out(self.weight)
            # gain = 1.0
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            M.init.uniform_(self.bias, -bound, bound)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        keys = ["weight"]
        if self.bias is not None:
            keys.append("bias")
        weights_to_load = {k: state_dict[prefix + k] for k in keys}
        for k, v in weights_to_load.items():
            if is_torch_tensor(v):
                weights_to_load[k] = v.detach().cpu().numpy()

        M.Module.load_state_dict(self, weights_to_load, strict)


Linear = patch_method(Linear, patch_override=False)
