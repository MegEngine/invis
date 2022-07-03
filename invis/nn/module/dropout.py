#!/usr/bin/env python3

from megengine.module import Dropout as MGE_Dropout

from .patch import patch_attribute, patch_method

__all__ = ["Dropout"]


class Dropout(MGE_Dropout):

    def __init__(self, p=0.5, inplace=False):
        kwargs = {"drop_prob": p}  # mge using `drop_prob` instead of `p`
        super().__init__(**kwargs)
        patch_attribute(self)


Dropout = patch_method(Dropout, patch_override=False)
