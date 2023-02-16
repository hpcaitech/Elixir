import contextlib
from typing import Dict, Iterator, Tuple

import torch

from elixir.tracer.utils import get_cuda_allocated, get_cuda_max_allocated


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def tensor_info(x: torch.Tensor):
    return (x.shape, x.stride(), x.layout, x.dtype)


def get_args_info(*args):
    info_list = []
    for x in args:
        if isinstance(x, torch.Tensor):
            info_list.append(tensor_info(x))
    return tuple(info_list)


class OpCache(object):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.temp_memory: Dict[Tuple, int] = dict()

    def get(self, info):
        if info in self.temp_memory:
            return True, self.temp_memory[info]
        else:
            return False, None

    def add(self, info, memo):
        self.temp_memory[info] = memo

    def print(self):
        print(f'OpCache {self.name} information:')
        for k, v in self.temp_memory.items():
            print(f'key: {k}\ntemp_memo:{v}')


aten = torch.ops.aten

ADDMM = OpCache('aten.addmm.default')
MM = OpCache('aten.mm.default')


def check_cuda(t):
    assert isinstance(t, torch.Tensor)
    assert t.device.type == 'cuda'


def get_output_shape(ma, mb):
    assert ma.dtype == mb.dtype
    assert ma.ndim == 2
    assert mb.ndim == 2

    u, v = ma.shape
    w, x = mb.shape

    assert v == w

    return (u, x)


def fake_cuda_output(temp_memo, output_shape, dtype):
    temp = torch.empty(temp_memo, dtype=torch.int8, device='cuda')
    del temp
    ret = torch.empty(output_shape, dtype=dtype, device='cuda')
    return ret


def real_cuda_output(func, *args, **kwargs):
    cur_alc = get_cuda_allocated()
    pre_max_alc = get_cuda_max_allocated()
    torch.cuda.reset_peak_memory_stats()

    with no_dispatch():
        ret = func(*args, **kwargs)

    max_alc = get_cuda_max_allocated()
    temp_memo = max_alc - cur_alc

    return ret, temp_memo, pre_max_alc


def fake_cuda_mm(*args, **kwargs):
    assert len(kwargs) == 0
    args_info = get_args_info(*args)
    cache_flag, temp_memo = MM.get(args_info)

    pre_max_alc = 0
    if cache_flag:
        ma, mb = args
        output_shape = get_output_shape(ma, mb)
        ret = fake_cuda_output(temp_memo, output_shape, ma.dtype)
    else:
        ret, temp_memo, pre_max_alc = real_cuda_output(aten.mm.default, *args, **kwargs)
        MM.add(args_info, temp_memo)

    return ret, pre_max_alc


def fake_cuda_addmm(*args, **kwargs):
    assert len(kwargs) == 0
    args_info = get_args_info(*args)
    cache_flag, temp_memo = ADDMM.get(args_info)

    pre_max_alc = 0
    if cache_flag:
        bias, ma, mb = args
        output_shape = get_output_shape(ma, mb)
        ret = fake_cuda_output(temp_memo, output_shape, ma.dtype)
    else:
        ret, temp_memo, pre_max_alc = real_cuda_output(aten.addmm.default, *args, **kwargs)
        ADDMM.add(args_info, temp_memo)

    return ret, pre_max_alc
