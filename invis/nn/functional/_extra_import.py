#!/usr/bin/env python3

import megengine.functional as F
from megengine.functional import *  # noqa

from invis.utils import get_import_names

from .inner_func import __all__ as _inner_func_names
from .redef_func import __all__ as _redef_func_names
from .special_func import __all__ as _special_func_names

_except_names = _redef_func_names + _inner_func_names + _special_func_names

__all__ = get_import_names(F, _except_names, import_type="function")

# from .redef_func import ensure_tensor_type
# def patch_mge_function(names):
#     for name in names:
#         f = getattr(F, name)
#         x = ensure_tensor_type(f)
#         import ipdb; ipdb.set_trace()
#         setattr(F, name, ensure_tensor_type(f))

# patch_mge_function(__all__)
