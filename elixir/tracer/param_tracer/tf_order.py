from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir import gpu_dev, no_dispatch, normalize_tuple
from elixir.parameter import is_no_hook_op
from elixir.tracer.utils import meta_copy


class FakeCudaTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        # TODO: clone strides and storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=gpu_dev(),
            requires_grad=elem.requires_grad)
        r.elem = elem.to('meta')
        return r

    def __repr__(self):
        return f'FCT({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def unwrap(x):
            return x.elem if isinstance(x, FakeCudaTensor) else x

        def wrap(x):
            return FakeCudaTensor(x) if isinstance(x, torch.Tensor) else x

        with no_dispatch():
            res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if len(res) == 1:
            return res[0]
        else:
            return res


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        Record.record_params(ctx.params)
        return (None, *grads)


class Record(FakeCudaTensor, nn.Parameter):
    record_steps: Dict = None

    def __new__(cls, elem):
        assert elem.device.type == 'meta', f'device type: {elem.device.type}'
        r = torch.Tensor._make_wrapper_subclass(cls,
                                                elem.size(),
                                                strides=elem.stride(),
                                                storage_offset=elem.storage_offset(),
                                                dtype=elem.dtype,
                                                layout=elem.layout,
                                                device=gpu_dev(),
                                                requires_grad=True)
        r.elem = elem
        return r

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if is_no_hook_op(func):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        params = list()

        def append_param(x):
            if isinstance(x, nn.Parameter):
                assert isinstance(x, Record)
                params.append(x)

        tree_map(append_param, args)
        tree_map(append_param, kwargs)
        Record.record_params(params)

        with torch._C.DisableTorchFunction():
            ret = normalize_tuple(func(*args, **kwargs))
            ret = PostFwdPreBwd.apply(params, *ret)

        def clone(t):
            if isinstance(t, torch.Tensor):
                t = t.clone()
            return t

        ret = tree_map(clone, ret)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    @staticmethod
    def reset():
        Record.record_steps = list()

    @staticmethod
    def steps():
        ret = Record.record_steps
        Record.record_steps = None
        return ret

    @staticmethod
    def record_params(params):
        record_dict = {p.param_name: p for p in params}
        Record.record_steps.append(record_dict)


def generate_tf_order(model: nn.Module, inp: Union[torch.Tensor, Tuple], step_fn: Callable):
    Record.reset()

    def mtensor_trans(t):
        meta_t = t.data.to('meta')
        if isinstance(t, nn.Parameter):
            meta_t = Record(meta_t)
        else:
            meta_t = FakeCudaTensor(meta_t)
        return meta_t

    model = meta_copy(model, mtensor_trans)
    for name, param in model.named_parameters():
        param.param_name = name

    def input_trans(t):
        if torch.is_tensor(t):
            meta_t = t.data.to('meta')
            meta_t.requires_grad = t.requires_grad
            meta_t = FakeCudaTensor(meta_t)
            return meta_t
        return t

    inp = normalize_tuple(inp)
    inp = tree_map(input_trans, inp)

    step_fn(model, *inp)

    ret = Record.steps()
    return ret
