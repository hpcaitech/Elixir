from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir.parameter import is_no_hook_op
from elixir.tracer.utils import meta_copy
from elixir.utils import no_dispatch, normalize_tuple


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

    def __new__(cls, elem):
        assert elem.device.type == 'meta', f'device type: {elem.device.type}'
        r = torch.Tensor._make_subclass(cls, elem)
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

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # notice: we should disable __torch_function__ here
        # otherwise, unexpected operations are called inside meta kernels
        with torch._C.DisableTorchFunction():
            with no_dispatch():
                return func(*args, **kwargs)

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
        record_dict = {p.param_name for p in params}
        Record.record_steps.append(record_dict)


def generate_tf_order(model: nn.Module, inp: Dict, step_fn: Callable):
    assert isinstance(inp, dict), 'The example input should be a dictionary'

    Record.reset()

    def mtensor_trans(t: torch.Tensor):
        meta_t = torch.empty_like(t, device='meta')
        if isinstance(t, nn.Parameter):
            meta_t = Record(meta_t)
            meta_t.requires_grad = t.requires_grad
        return meta_t

    model = meta_copy(model, mtensor_trans)
    for name, param in model.named_parameters():
        param.param_name = name

    def input_trans(t):
        if torch.is_tensor(t):
            meta_t = torch.empty_like(t, device='meta', requires_grad=t.requires_grad)
            return meta_t
        return t

    inp = tree_map(input_trans, inp)

    step_fn(model, inp)

    ret = Record.steps()
    return ret
