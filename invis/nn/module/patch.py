#!/usr/bin/env python3

import inspect
import itertools
from collections import OrderedDict
from types import FunctionType
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, Union

import megengine as mge
from megengine.module import Module

from invis.nn.functional.helper import ensure_tensor_type
from invis.utils.wrapper import is_torch_tensor

__all__ = ["ModuleMixin"]


class ModuleMixin:
    """
    used to patch module behaviour of MegEngine module

    NOTE: to align with torch, some code is copied from torch.nn.Module and then modified.
    """

    def add_module(self: Module, name: str, module):
        r"""Adds a child module to the current module.
        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        def typename(obj):
            return obj.__class__.__name__

        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{typename(module)} is not a Module subclass")
        elif not isinstance(name, (str, bytes)):
            raise TypeError(f"module name should be a string. Got {typename(module)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")

        Module.__setattr__(self, name, module)

    def load_state_dict(self: Module, state_dict, strict: bool = True):
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        if callable(state_dict):
            loaded_keys, missing_keys = self._load_state_dict_with_closure(state_dict)
            return

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()

        def load(module, prefix=''):
            local_metadata = {}
            if hasattr(module, '_load_from_state_dict'):
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True,
                    missing_keys, unexpected_keys, error_msgs
                )

            for name, child in module.named_children():
                if isinstance(child, Module):
                    load(child, prefix + name + '.')  # noqa

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))

    def register_parameter(self, name: str, param: Optional[mge.Parameter]) -> None:
        r"""Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. ")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, mge.Parameter):
            raise TypeError(f"cannot assign object to parameter '{name}' "
                            "(invis.nn.Parameter or None required)")
        else:
            self._parameters[name] = param

    def register_buffer(self: Module, name: str, tensor, persistent: bool = True) -> None:
        r"""Adds a buffer to the module.
        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.
        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        if '_buffers' not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. ")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, mge.Tensor):
            raise TypeError(f"cannot assign object to buffer '{name}' "
                            "(mge Tensor or None required)")
        else:
            self._buffers[name] = tensor
            # TODO persistent is not used in this.

    def _load_from_state_dict(
        self: Module, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        """
        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        persistent_buffers = {k: v for k, v in self._buffers.items()}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if is_torch_tensor(input_param):
                    input_param = input_param.cpu().detach().numpy()
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '  # noqa
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue
                try:
                    param._reset(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}, '
                        'an exception occurred : {}.'
                        .format(key, param.size(), input_param.size(), ex.args)
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def to(self: Module, *args, **kwargs):
        return self

    def eval(self: Module):
        # eval in megengine returns nothing, stupid
        Module.eval(self)
        return self

    def _flatten(
        self: Module,
        *,
        recursive: bool = True,
        with_key: bool = False,
        with_parent: bool = False,
        prefix: Optional[str] = None,
        predicate: Callable[[Any], bool] = lambda _: True,
        seen: Optional[Set[int]] = None
    ) -> Union[Iterable[Any], Iterable[Tuple[str, Any]]]:
        # NOTE: I don't like design here in megengine, so I override it
        # 1. MGE using expand sturcture
        # 2. sorted(module_dict) in origin code, which means "bn" is solved before "conv"
        if seen is None:
            seen = set([id(self)])

        _prefix = "" if prefix is None else prefix + "."
        module_dict = vars(self)

        def filter_value(prefix):
            obj = None
            if prefix in module_dict:
                obj = module_dict[prefix]
            else:
                if hasattr(self, "_parameters") and prefix in self._parameters:
                    obj = self._parameters[prefix]
                elif hasattr(self, "_buffers") and prefix in self._buffers:
                    obj = self._buffers[prefix]

            if isinstance(obj, (mge.Tensor, Module)):
                return [(prefix, obj)]
            else:
                return []

        iter_keys = self._modules.copy()
        _prefix = "" if prefix is None else prefix + "."

        if hasattr(self, "_parameters"):
            iter_keys.extend(list(self._parameters.keys()))
        if hasattr(self, "_buffers"):
            iter_keys.extend(list(self._buffers.keys()))

        extra_keys = [x for x in list(module_dict.keys()) if x not in iter_keys]
        iter_keys.extend(extra_keys)

        for key in iter_keys:
            for expanded_key, leaf in filter_value(key):
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)

                if predicate(leaf):
                    if with_key and with_parent:
                        yield _prefix + expanded_key, leaf, self
                    elif with_key:
                        yield _prefix + expanded_key, leaf
                    elif with_parent:
                        yield leaf, self
                    else:
                        yield leaf

                if recursive and isinstance(leaf, Module):
                    yield from leaf._flatten(
                        recursive=recursive,
                        with_key=with_key,
                        with_parent=with_parent,
                        prefix=_prefix + expanded_key if with_key else None,
                        predicate=predicate,
                        seen=seen,
                    )

    def __setattr__(self: Module, name: str, value: Any) -> None:
        if not isinstance(value, Module) and isinstance(value, (list, tuple, dict)):
            # support List as ModuleList and Dict as ModuleDict is stupaid for me in MegEngine
            # I fucking hate it...
            object.__setattr__(self, name, value)
        else:
            Module.__setattr__(self, name, value)


def patch_method(
    classname,
    patch_obj=ModuleMixin,
    skip: Iterable[str] = None,
    patch_override: bool = True,
    patch_forward: bool = True,
):
    """patch method in `patch_obj` to another class

    Args:
        classname: class to patch.
        patch_obj: class object to patch. Defaults to ModuleMixin.
        skip (Iterable[str], optional): skip method list. Defaults to None.
        patch_override (bool, optional): patch override method in origin class. Defaults to True.
        patch_forwad (bool, optional): patch forward to ensure tensor output.

    Returns:
        new class with method patched.
    """
    if skip is None:
        skip = []
    if not patch_override:
        from invis.utils.helper import get_override_functions
        skip.extend(get_override_functions(classname))

    members = {m[0]: m[1] for m in inspect.getmembers(patch_obj) if isinstance(m[1], FunctionType)}
    for name, func in members.items():
        if name not in skip:
            setattr(classname, name, func)

    if patch_forward:
        classname.forward = ensure_tensor_type(classname.forward)

    return classname


def patch_attribute(obj):
    obj._buffers = OrderedDict()
    obj._parameters = OrderedDict()
