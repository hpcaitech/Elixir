from collections import OrderedDict

import torch
import torch.nn as nn
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
        return (None, *args)

    @staticmethod
    def backward(ctx, *grads):
        return (None, *grads)


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        for p in ctx.params:
            p.data = p.fake_data
        return (None, *args)

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
        params_index = 1

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
            for x in args:
                print('args', type(x))
            for y in kwargs:
                print('kwargs', type(y))

            ret = func(*tree_map(replace_param, args), **tree_map(replace_param, kwargs))
        assert not isinstance(ret, tuple)
        with torch._C.DisableTorchFunction():
            ret = PostFwdPreBwd.apply(params, ret)

        ret = ret[1]
        ret = ret[:]

        assert isinstance(ret, torch.Tensor)

        # if len(ret) == 1:
        #     ret = ret[0]

        return ret
