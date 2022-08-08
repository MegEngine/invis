#!/usr/bin/env python3

import math

import megengine.module as M
from megengine.module import Linear as MGE_Linear

from .patch import patch_method

__all__ = ["Linear"]


class Linear(MGE_Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        M.init.msra_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = M.init.calculate_fan_in_and_fan_out(self.weight)
            # gain = 1.0
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            M.init.uniform_(self.bias, -bound, bound)


Linear = patch_method(Linear, patch_override=False)
