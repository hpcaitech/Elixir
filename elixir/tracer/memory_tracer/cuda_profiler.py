from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir.tracer.utils import meta_copy

from .memory_tensor import MTensor


def cuda_memory_profiling(model: nn.Module, inp: Union[torch.Tensor, Tuple], step_fn: Callable, dtype=torch.float):

    print(f'You are profiling cuda memory with dtype `{dtype}`')

    def tensor_trans(t):
        meta_t = t.data.to('meta')
        if isinstance(t, nn.Parameter):
            meta_t = nn.Parameter(meta_t.to(dtype))
        return meta_t

    model = meta_copy(model, tensor_trans)

    param_occ = 0
    max_numel = 0
    for name, param in model.named_parameters():
        assert param.dtype is dtype
        param_occ += param.numel() * param.element_size()
        max_numel = max(max_numel, param.numel())

    buffer_occ = 0
    for name, buffer in model.named_buffers():
        buffer_occ += buffer.numel() * buffer.element_size()

    # get the initial cuda memory allocation for sanity check
    init_cuda_alc = torch.cuda.memory_allocated()

    pool = torch.empty(max_numel, dtype=dtype, device='cuda')

    def tensor_to_cuda(t):
        if isinstance(t, nn.Parameter):
            fake_data = pool[:t.numel()].view(t.shape)
            return nn.Parameter(fake_data)
        else:
            fake_data = torch.empty(t.shape, dtype=t.dtype, device='cuda')
            return fake_data

    model = meta_copy(model, tensor_to_cuda)

    # convert all input data to meta_tensor
    if not isinstance(inp, tuple):
        inp = (inp,)
    inp = tree_map(lambda t: MTensor(t.data.to('cuda')), inp)

    torch.cuda.reset_peak_memory_stats()
    before_cuda_alc = torch.cuda.memory_allocated()
    step_fn(model, inp)
    after_cuda_alc = torch.cuda.max_memory_allocated()

    activation_occ = after_cuda_alc - before_cuda_alc

    del inp
    del model
    del pool
    close_cuda_alc = torch.cuda.memory_allocated()
    assert init_cuda_alc == close_cuda_alc

    return dict(param_occ=param_occ, buffer_occ=buffer_occ, grad_occ=param_occ, activation_occ=activation_occ)
