from collections import OrderedDict
from copy import copy

import torch
import torch.nn as nn
from torch.fx.immutable_collections import immutable_dict
from torch.utils._pytree import tree_map

from elixir import calc_buffer_size, gpu_dev
from elixir.parameter import FakeTensor, OutplaceTensor, is_no_hook_op, to_outplace_tensor


class Store(object):

    def __init__(self, buffer_size: torch.Tensor, buffer_dtype: torch.dtype) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer_dtype = buffer_dtype
        self.buffer: torch.Tensor = torch.empty(buffer_size, dtype=buffer_dtype, device=gpu_dev())
        self.record_dict = dict()

    def insert(self, t: torch.Tensor, offset: int) -> int:
        assert t not in self.record_dict
        end = offset + t.numel()
        assert end <= self.buffer_size, f'buffer size is {self.buffer_size} but needs {end}'

        new_data = self.buffer[offset:end].view(t.shape)
        new_data.copy_(t.data)

        self.record_dict[t] = t.data
        t.data = new_data

        return end

    def erase(self, t: torch.Tensor):
        assert t in self.record_dict

        new_data = self.record_dict.pop(t)
        t.data = new_data

        return


def prefwd_postbwd_func(store: Store):

    class PreFwdPostBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            with torch._C.DisableTorchFunction():
                ctx.params = params
                for p in ctx.params:
                    p.data = p.my_data

                offset = 0
                for p in ctx.params:
                    offset = store.insert(p, offset)
                return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                for p in ctx.params:
                    torch.zero_(p.data)
                    store.erase(p)
                return (None, *grads)

    return PreFwdPostBwd.apply


def postfwd_prebwd_func(store: Store):

    class PostFwdPreBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, name, *args):
            with torch._C.DisableTorchFunction():
                ctx.params = params
                ctx.name = name
                for p in ctx.params:
                    torch.zero_(p.data)
                    store.erase(p)
                return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                # print("backward name", ctx.name)
                for p in ctx.params:
                    p.data = p.my_data

                offset = 0
                for p in ctx.params:
                    offset = store.insert(p, offset)
                return (None, None, *grads)

    return PostFwdPreBwd.apply


class MyParameter(OutplaceTensor, nn.Parameter):

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

    @staticmethod
    def attach_functions(store: Store):
        MyParameter.pre_post = prefwd_postbwd_func(store)
        MyParameter.post_pre = postfwd_prebwd_func(store)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if is_no_hook_op(func):
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
        new_params = MyParameter.pre_post(params, *params)

        def replace_param(x):
            if isinstance(x, MyParameter):
                return new_params[params_to_index[x]]
            return x

        with torch._C.DisableTorchFunction():
            # for x in new_params:
            #     print("new_params", type(x))
            #     if isinstance(x, torch.Tensor):
            #         print(x.shape, x.dtype)
            # for x in args:
            #     print("args", type(x))
            #     if isinstance(x, torch.Tensor):
            #         print(x.shape, x.dtype)
            #
            # for k, v in kwargs.items():
            #     print("kwargs", k, v)
            #     if isinstance(v, torch.Tensor):
            #         print(v.shape, v.dtype)

            ret = func(*tree_map(replace_param, args), **tree_map(replace_param, kwargs))

        if not isinstance(ret, tuple):
            ret = (ret,)

        ptr_set = set()
        for p in new_params:
            ptr_set.add(p.data_ptr())

        def clone_inplace_tensor(x):
            if isinstance(x, torch.Tensor):
                start_point = x.data_ptr() - x.element_size() * x.storage_offset()
                if start_point in ptr_set:
                    return x.clone()
            return x

        ret = tree_map(clone_inplace_tensor, ret)
        ret = MyParameter.post_pre(params, func.__name__, *ret)

        def convert(t):
            if isinstance(t, torch.Tensor):
                t = to_outplace_tensor(t)
            return t

        ret = tree_map(convert, ret)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


def transform(m: nn.Module) -> nn.Module:
    buffer_size = calc_buffer_size(m)
    MyParameter.attach_functions(Store(buffer_size, torch.float))

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

    def transform_input(self_module, inputs):
        input_list = list()
        for t in inputs:
            if isinstance(t, torch.Tensor):
                t = OutplaceTensor(t)
            input_list.append(t)
        return tuple(input_list)

    m.register_forward_pre_hook(transform_input)

    return m


def main():
    x = torch.randn(4, 4, requires_grad=True)
    z = OutplaceTensor(x)

    def my_func(x, y):
        return torch.add(x, y)

    res = my_func(z, z)
    print(type(res))


if __name__ == '__main__':
    main()
