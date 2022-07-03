#!/usr/bin/env python3
import inspect

__all__ = [
    "get_import_names",
    "get_override_functions",
]


def is_private_attr(name: str):
    return name.startswith("__") and name.endswith("__")


def is_protected_attr(name: str):
    return name.startswith("_") and name.endswith("_")


def get_import_names(module, except_names=None, import_type="class"):
    """
    Args:
        import_type: one of "class", "function" and "module", otherwise filter nothing.
    """
    if except_names is None:
        except_names = list()

    if import_type == "class":
        f = inspect.isclass
    elif import_type == "function":
        f = inspect.isfunction
    elif import_type == "module":
        f = inspect.ismodule
    else:
        f = lambda x: x is not None  # noqa

    names = [x for x in dir(module) if not is_private_attr(x) and x not in except_names]
    names = [x for x in names if f(getattr(module, x, None))]
    return names


def get_override_functions(class_name):
    names = [k for k, v in vars(class_name).items() if callable(v)]
    return names
