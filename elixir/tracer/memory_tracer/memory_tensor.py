import contextlib
from typing import Iterator

import torch
from torch.utils._pytree import tree_map

from elixir.tracer.utils import get_cuda_allocated, get_cuda_max_allocated

from .op_cache import fake_cuda_addmm, fake_cuda_mm

aten = torch.ops.aten


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class MTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    peak_memory_allocated: int = 0

    @staticmethod
    def reset_peak_memory():
        MTensor.peak_memory_allocated = 0

    @staticmethod
    def update_peak_memory(new_peak):
        MTensor.peak_memory_allocated = max(MTensor.peak_memory_allocated, new_peak)

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
            device=elem.device,
            requires_grad=elem.requires_grad)
        r.elem = elem
        return r

    def __repr__(self):
        return f'MTensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def print_tensor(x):
            if isinstance(x, torch.Tensor):
                print(x.shape)

        # tree_map(print_tensor, args)
        # tree_map(print_tensor, kwargs)

        def unwrap(x):
            return x.elem if isinstance(x, MTensor) else x

        def wrap(x):
            return MTensor(x) if isinstance(x, torch.Tensor) else x

        if func is aten.addmm.default:
            # res = fake_cuda_addmm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            print('addmm pre', get_cuda_allocated(), get_cuda_max_allocated())
            res, pre_max = fake_cuda_addmm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            MTensor.update_peak_memory(pre_max)
            print('addmm aft', get_cuda_allocated(), get_cuda_max_allocated())
        elif func is aten.mm.default:
            # res = fake_cuda_mm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            print('mm pre', get_cuda_allocated())
            res, pre_max = fake_cuda_mm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            MTensor.update_peak_memory(pre_max)
            print('mm aft', get_cuda_allocated())
        else:
            with no_dispatch():
                res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        # with no_dispatch():
        #     res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if len(res) == 1:
            return res[0]
        else:
            return res
