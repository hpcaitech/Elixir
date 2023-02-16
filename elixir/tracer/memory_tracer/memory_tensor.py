import contextlib
from typing import Iterator

import torch
from torch.utils._pytree import tree_map

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


def check_cuda(t):
    assert isinstance(t, torch.Tensor)
    assert t.device.type == 'cuda'


def fake_cuda_mm(ma, mb):
    check_cuda(ma)
    check_cuda(mb)

    assert ma.dtype == mb.dtype
    assert ma.ndim == 2
    assert mb.ndim == 2

    u, v = ma.shape
    w, x = mb.shape

    assert v == w

    return torch.empty((u, x), dtype=ma.dtype, device='cuda')


def fake_cuda_addmm(bias, ma, mb):
    check_cuda(bias)
    assert bias.dtype == ma.dtype
    return fake_cuda_mm(ma, mb)


class MTensor(torch.Tensor):
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
            res = fake_cuda_addmm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        elif func is aten.mm.default:
            res = fake_cuda_mm(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        else:
            with no_dispatch():
                res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if len(res) == 1:
            return res[0]
        else:
            return res
