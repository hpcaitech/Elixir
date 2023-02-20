import gc
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir.tracer.utils import get_cuda_allocated, meta_copy, model_memory_figure

from .memory_tensor import MTensor


def grad_cleaner(grad):
    empty_grad = torch.empty_like(grad.elem)
    grad.elem = None
    empty_grad.storage().resize_(0)
    return empty_grad


def cuda_memory_profiling(model: nn.Module, inp: Union[torch.Tensor, Tuple], step_fn: Callable, dtype=torch.float):

    print(f'You are profiling cuda memory with dtype `{dtype}`')

    def tensor_trans(t):
        meta_t = t.data.to('meta')
        if isinstance(t, nn.Parameter):
            meta_t = nn.Parameter(meta_t.to(dtype))
        return meta_t

    # first, transform the model into one dtype
    model = meta_copy(model, tensor_trans)
    # get the memory firgure of the model
    memo_dict = model_memory_figure(model)
    # initialize a empty pool for parameters
    pool = torch.empty(memo_dict['param_max_numel'], dtype=dtype, device='cuda')

    def tensor_to_cuda(t):
        if isinstance(t, nn.Parameter):
            fake_data = pool[:t.numel()].view(t.shape)
            return nn.Parameter(fake_data)
        else:
            fake_data = torch.empty(t.shape, dtype=t.dtype, device='cuda')
            return fake_data

    # make all parameters in CUDA and point to a same address
    model = meta_copy(model, tensor_to_cuda)
    # add hooks to clean gradients
    for param in model.parameters():
        param.register_hook(grad_cleaner)
    # convert all input data to meta_tensor
    if not isinstance(inp, tuple):
        inp = (inp,)
    inp = tree_map(lambda t: MTensor(t.cuda()), inp)
    # reset all collected peak memory states
    MTensor.reset_peak_memory()
    before_cuda_alc = get_cuda_allocated()

    step_fn(model, inp)

    after_cuda_alc = MTensor.current_peak_memory()
    activation_occ = after_cuda_alc - before_cuda_alc

    return dict(param_occ=memo_dict['param_occ'],
                buffer_occ=memo_dict['buffer_occ'],
                grad_occ=memo_dict['param_occ'],
                activation_occ=activation_occ)
