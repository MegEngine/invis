#!/usr/bin/env python3

from megengine.module import Module as MGE_Module

from .patch import patch_attribute, patch_method


class Module(MGE_Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)


Module = patch_method(Module, patch_override=False)
