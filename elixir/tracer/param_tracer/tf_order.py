from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir import normalize_tuple
from elixir.parameter import is_tensor_output
from elixir.tracer.utils import meta_copy


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        Record.record_params(ctx.params)
        return (None, *grads)


class Record(nn.Parameter):
    record_steps: Dict = None

    def __new__(cls, param):
        assert param.device.type == 'meta'
        r = torch.Tensor._make_subclass(cls, param)
        return r

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if is_tensor_output(func):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        params = list()

        def append_param(x):
            if isinstance(x, nn.Parameter):
                assert isinstance(x, Record)
                params.appen(x)

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
        record_dict = {p.name: p for p in params}
        Record.record_steps.append(record_dict)


def generate_td_order(model: nn.Module, inp: Union[torch.Tensor, Tuple], step_fn: Callable):
    Record.reset()

    def mtensor_trans(t):
        meta_t = t.data.to('meta')
        if isinstance(t, nn.Parameter):
            meta_t = Record(meta_t)
        return meta_t

    model = meta_copy(model, mtensor_trans)
    for name, param in model.named_parameters():
        param.name = name

    def input_trans(t):
        if torch.is_tensor(t):
            meta_t = t.data.to('meta')
            meta_t.requires_grad = t.requires_grad
            return meta_t
        return t

    inp = normalize_tuple(inp)
    inp = tree_map(input_trans, inp)

    step_fn(model, *inp)

    ret = Record.steps()
    return ret
