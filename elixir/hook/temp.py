from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkFetcher, ChunkGroup, FIFOScheduler, MemoryPool, TensorState
from elixir.parameter import OutplaceTensor

from .parameter import HookParam


def to_divide(a: int, b: int):
    return a + (-a % b)


def grad_handler(grad: torch.Tensor, param: nn.Parameter, fetcher: ChunkFetcher):
    empty_grad = torch.empty_like(grad)
    empty_grad.storage().resize_(0)

    with torch._C.DisableTorchFunction():
        chunk = fetcher.get_chunks([param])[0]
        if chunk.tensors_info[param].state != TensorState.HOLD_AFTER_BWD:
            raise RuntimeError()
        fetcher.group.tensor_trans_state(param, TensorState.READY_FOR_REDUCE)
        chunk.copy_tensor_to_chunk_slice(param, grad)
        fetcher.reduce_chunk(chunk)

    return empty_grad


def hook_transform(model: nn.Module, process_group: dist.ProcessGroupGloo):
    pg_size = dist.get_world_size(process_group)

    private_list = list()
    for param in model.parameters():
        param.__class__ = HookParam
        private_list.append(BlockRequire(param.numel(), param.dtype))

    mp = MemoryPool('cuda')
    mp.allocate(private_block_list=private_list)
    cg = ChunkGroup(rcache=mp)
    # allocate chunk group
    fused_config = dict(rcache_fused=True)
    for param in model.parameters():
        cg.allocate_chunk([param], to_divide(param.numel(), pg_size), param.dtype, process_group, fused_config)
    # initialize chunk fetcher
    scheduler = FIFOScheduler()
    fetcher = ChunkFetcher(scheduler, cg)
    # register fetcher and gradient handler
    HookParam.attach_fetcher(fetcher)
    for param in model.parameters():
        param.register_hook(partial(grad_handler, param=param, fetcher=fetcher))
    # set inplace to False for all modules
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    def transform_input(self_module, inputs):
        input_list = list()
        for t in inputs:
            if isinstance(t, torch.Tensor):
                t = OutplaceTensor(t)
            input_list.append(t)
        return tuple(input_list)

    model.register_forward_pre_hook(transform_input)

    return model
