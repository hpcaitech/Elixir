from collections import OrderedDict
from copy import copy
from typing import Optional, Set

import torch.nn as nn


def _get_dfs_module_list(module: nn.Module, memo: Optional[Set[nn.Module]] = None, prefix: str = ''):
    """Get a dfs module list of the given module. Its order is same as the order of creations of modules.
    """
    if memo is None:
        memo = set()
    if module not in memo:
        for name, submodule in module._modules.items():
            if submodule is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in _get_dfs_module_list(submodule, memo, submodule_prefix):
                yield m

        memo.add(module)
        yield prefix, module


def _get_shallow_copy_model(model: nn.Module):
    """Get a shallow copy of the given model. Each submodule is different from the original submodule.
    But the new submodule and the old submodule share all attributes.
    """
    old_to_new = dict()
    for name, module in _get_dfs_module_list(model):
        new_module = copy(module)
        new_module._modules = OrderedDict()
        for subname, submodule in module._modules.items():
            if submodule is None:
                continue
            setattr(new_module, subname, old_to_new[submodule])
        old_to_new[module] = new_module
    return old_to_new[model]


def meta_copy(model: nn.Module):
    new_model = _get_shallow_copy_model(model)

    for (_, old_module), (_, new_module) in \
        zip(_get_dfs_module_list(model), _get_dfs_module_list(new_model)):

        new_module._parameters = OrderedDict()
        for name, param in old_module.named_parameters(recurse=False):
            new_param = None
            if param is not None:
                new_param = nn.Parameter(param.data.to('meta'))
            setattr(new_module, name, new_param)

        new_module._buffers = OrderedDict()
        for name, buffer in old_module.named_buffers(recurse=False):
            new_buffer = None
            if buffer is not None:
                new_buffer = buffer.to('meta')
            setattr(new_module, name, new_buffer)

    return new_model
