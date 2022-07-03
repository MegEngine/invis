#!/usr/bin/env python3

# This file is copied from pytorch and modified.
import operator
import warnings
from collections import OrderedDict
from collections import abc as container_abcs
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union

import megengine as mge
import megengine.module as M
from megengine import Parameter

import invis as torch

from .module import Module
from .patch import ModuleMixin, patch_attribute, patch_method


class Container(Module):

    def __init__(self, **kwargs: Any) -> None:
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(M.Sequential):
    """A sequential container."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_attribute(self)

    def add_module(self, name: str, module: M.Module):
        ModuleMixin.add_module(self, name, module)
        self.layer_keys.append(name)


Sequential = patch_method(Sequential, patch_override=False)


class ModuleList(Module):
    r"""Holds submodules in a list.
    :class:`~invis.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~invis.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::
        class MyModule(nn.Module):

            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """
    # _inner_modules: Dict[str, Module]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()
        self._inner_modules = OrderedDict()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def add_module(self, name: str, module: Module) -> None:
        ModuleMixin.add_module(self, name, module)
        self._inner_modules[name] = module

    def __getitem__(self, idx: int) -> Module:
        if isinstance(idx, slice):
            return self.__class__(list(self._inner_modules.values())[idx])
        else:
            return self._inner_modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._inner_modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._inner_modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._inner_modules.values())))

    def __len__(self) -> int:
        return len(self._inner_modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._inner_modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> 'ModuleList':
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.
        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._inner_modules), index, -1):
            self._inner_modules[str(i)] = self._inner_modules[str(i - 1)]
        self._inner_modules[str(index)] = module

    def append(self, module: Module) -> 'ModuleList':
        r"""Appends a given module to the end of the list.
        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        r"""Appends modules from a Python iterable to the end of the list.
        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def forward(self):
        raise NotImplementedError()


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        self.inner_modules = {}  # pytorch use _modules, while we have already use
        self._modules = []  # make _load_state_dict work
        if modules is not None:
            self.update(modules)

    def add_module(self, name: str, module: M.Module):
        self.inner_modules[name] = module
        self._modules.append(name)

    def __getitem__(self, key: str) -> Module:
        return self.inner_modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self.inner_modules[key]

    def __len__(self) -> int:
        return len(self.inner_modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner_modules)

    def __contains__(self, key: str) -> bool:
        return key in self.inner_modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self.inner_modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        return self.inner_modules.keys()

    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self.inner_modules.items()

    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self.inner_modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at),
                # or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    def forward(self):
        raise NotImplementedError()

    def _flatten(
        self,
        *,
        recursive: bool = True,
        with_key: bool = False,
        with_parent: bool = False,
        prefix: Optional[str] = None,
        predicate: Callable[[Any], bool] = lambda _: True,
        seen: Optional[Set[int]] = None
    ) -> Union[Iterable[Any], Iterable[Tuple[str, Any]]]:
        # TODO move this to mixin
        if seen is None:
            seen = set([id(self)])

        _prefix = "" if prefix is None else prefix + "."

        for key, leaf in self.inner_modules.items():
            if isinstance(leaf, M.Module):
                expanded_key = key
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

                if recursive and isinstance(leaf, M.Module):
                    yield from leaf._flatten(
                        recursive=recursive,
                        with_key=with_key,
                        with_parent=with_parent,
                        prefix=_prefix + expanded_key if with_key else None,
                        predicate=predicate,
                        seen=seen,
                    )

    def _state_dict(self, rst=None, prefix="", keep_var=False):
        r"""Returns a dictionary containing whole states of the module."""
        module_type = self.__class__
        if rst is None:
            rst = OrderedDict()

        for k, v in self._flatten(
            recursive=False, with_key=True, predicate=lambda x: isinstance(x, mge.Parameter)
        ):
            assert prefix + k not in rst, "duplicated state: {}".format(k)
            rst[(module_type, prefix + k)] = v if keep_var else v.numpy()

        for k, submodule in self._flatten(
            recursive=False,
            with_key=True,
            predicate=lambda obj: isinstance(obj, M.Module),
        ):
            submodule.state_dict(rst, prefix + k + ".", keep_var)

        return rst

    def __repr__(self):
        def add_indent(repr_str, num_spaces):
            s = repr_str.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return repr_str
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        extra_lines = []
        extra_repr = self._module_info_string()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for name, module in self.items():
            if isinstance(module, M.Module):
                child_lines.append(
                    "(" + name + "): " + add_indent(repr(module), 2)
                )

        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class ParameterList(Module):
    r"""Holds parameters in a list.
    :class:`~torch.nn.ParameterList` can be indexed like a regular Python
    list, but parameters it contains are properly registered, and will be
    visible by all :class:`~torch.nn.Module` methods.
    Args:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter` to add
    Example::
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])  # noqa
            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters: Optional[Iterable['Parameter']] = None) -> None:
        super(ParameterList, self).__init__()
        self._initialized = True
        if parameters is not None:
            self += parameters

    def __setstate__(self, state):
        state['_initialized'] = False
        super(ParameterList, self).__setstate__(state)
        self._initialized = True

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._parameters.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._parameters[str(idx)]

    def __setitem__(self, idx: int, param: 'Parameter') -> None:
        idx = self._get_abs_string_index(idx)
        return self.register_parameter(str(idx), param)

    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and not isinstance(value, torch.nn.Parameter):
                warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator['Parameter']:
        return iter(self._parameters.values())

    def __iadd__(self, parameters: Iterable['Parameter']) -> 'ParameterList':
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter: 'Parameter') -> 'ParameterList':
        """Appends a given parameter at the end of the list.
        Args:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable['Parameter']) -> 'ParameterList':
        """Appends parameters from a Python iterable to the end of the list.
        Args:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn("nn.ParameterList is being used with DataParallel but this is not "
                      "supported. This list will appear empty for the models replicated "
                      "on each GPU except the original one.")

        return super(ParameterList, self)._replicate_for_data_parallel()


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.
    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.
    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary that respects
    * the order of insertion, and
    * in :meth:`~torch.nn.ParameterDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ParameterDict` (the argument to
      :meth:`~torch.nn.ParameterDict.update`).
    Note that :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.
    Args:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter`) or an iterable of key-value pairs
            of type (string, :class:`~torch.nn.Parameter`)
    Example::
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })
            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Optional[Mapping[str, 'Parameter']] = None) -> None:
        super(ParameterDict, self).__init__()
        self._initialized = True
        if parameters is not None:
            self.update(parameters)

    def __setstate__(self, state):
        state['_initialized'] = False
        super(ParameterDict, self).__setstate__(state)
        self._initialized = True

    def __getitem__(self, key: str) -> 'Parameter':
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: 'Parameter') -> None:
        self.register_parameter(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and not isinstance(value, torch.nn.Parameter):
                warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def clear(self) -> None:
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()

    def pop(self, key: str) -> 'Parameter':
        r"""Remove key from the ParameterDict and return its parameter.
        Args:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()

    def items(self) -> Iterable[Tuple[str, 'Parameter']]:
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self) -> Iterable['Parameter']:
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters: Mapping[str, 'Parameter']) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.
        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.
        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                self[p[0]] = p[1]  # type: ignore[assignment]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn("nn.ParameterDict is being used with DataParallel but this is not "
                      "supported. This dict will appear empty for the models replicated "
                      "on each GPU except the original one.")

        return super(ParameterDict, self)._replicate_for_data_parallel()
