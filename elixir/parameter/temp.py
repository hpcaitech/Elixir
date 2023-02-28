from collections import OrderedDict
from copy import copy

import torch
import torch.nn as nn
from torch.fx.immutable_collections import immutable_dict
from torch.utils._pytree import tree_map


class FakeTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(cls,
                                                elem.size(),
                                                strides=elem.stride(),
                                                storage_offset=elem.storage_offset(),
                                                dtype=elem.dtype,
                                                layout=elem.layout,
                                                device=elem.device,
                                                requires_grad=elem.requires_grad)
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError


class PreFwdPostBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        for p in ctx.params:
            p.data = p.my_data
        return args

    @staticmethod
    def backward(ctx, *grads):
        return (None, *grads)


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        for p in ctx.params:
            p.data = p.fake_data
        return args

    @staticmethod
    def backward(ctx, *grads):
        for p in ctx.params:
            p.data = p.my_data
        return (None, *grads)


class MyParameter(nn.Parameter):

    def __new__(cls, tensor, requires_grad=True):
        r = torch.Tensor._make_subclass(cls, tensor, require_grad=requires_grad)
        with torch._C.DisableTorchFunction():
            r.my_shape = tensor.shape
            r.my_dtype = tensor.dtype
            r.my_device = tensor.device
            r.my_data = r.data
            r.fake_data = FakeTensor(r.my_data)
            r.data = r.fake_data
        return r

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__.startswith('__'):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        params_to_index = OrderedDict()
        params_index = 0

        def append_param(x):
            nonlocal params_index
            if isinstance(x, MyParameter):
                params_to_index[x] = params_index
                params_index += 1

        tree_map(append_param, args)
        tree_map(append_param, kwargs)

        params = tuple(params_to_index.keys())
        with torch._C.DisableTorchFunction():
            new_params = PreFwdPostBwd.apply(params, *params)

        def replace_param(x):
            if isinstance(x, MyParameter):
                return new_params[params_to_index[x]]
            return x

        with torch._C.DisableTorchFunction():
            ret = func(*tree_map(replace_param, args), **tree_map(replace_param, kwargs))

        with torch._C.DisableTorchFunction():
            ret = PostFwdPreBwd.apply(params, ret)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


def transform(m: nn.Module) -> nn.Module:
    # transform each parameter to MyParameter
    for m_name, module in m.named_modules():
        param_list = list(module.named_parameters(recurse=False))
        for p_name, param in param_list:
            new_param = MyParameter(param.data)
            delattr(module, p_name)
            setattr(module, p_name, new_param)

    # set inplace to False for all modules
    for module in m.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
    for node in gm.graph.nodes:
        if node.op in ('call_function', 'call_method'):
            if 'inplace' in node.kwargs:
                new_kwargs = {k: v for k, v in node.kwargs.items() if k != 'inplace'}
                node.kwargs = immutable_dict(new_kwargs)

    # remove inplace operations
    gm.recompile()

    # print(gm.graph)
    # exit(0)

    return gm
