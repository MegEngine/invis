#!/usr/bin/env python3

import megengine.module as M

from .patch import patch_attribute, patch_method

__all__ = ["LayerNorm", "GroupNorm"]


class LayerNorm(M.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)


LayerNorm = patch_method(LayerNorm, patch_override=False)


class GroupNorm(M.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)


GroupNorm = patch_method(GroupNorm, patch_override=False)
