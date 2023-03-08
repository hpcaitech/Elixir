from functools import partial

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from elixir.chunk import ChunkFetcher, TensorState
from elixir.hook import HookParam
from elixir.parameter import OutplaceTensor


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


class ElixirModel(nn.Module):

    def __init__(self, module: nn.Module, fetcher: ChunkFetcher) -> None:
        super().__init__()
        self.module = module
        self.fetcher = fetcher

        self.set_module_outplace(module)
        for param in module.parameters():
            param.register_hook(partial(grad_handler, param=param, fetcher=fetcher))
            param.__class__ = HookParam

    def set_module_outplace(m: nn.Module):
        # set inplace to False for all modules
        for module in m.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

    def forward(self, *args, **kwargs):
        self.fetcher.reset()
        HookParam.attach_fetcher(self.fetcher)

        def to_outplace_tensor(t):
            if torch.is_tensor(t):
                t = OutplaceTensor(t)
            return t

        args = tree_map(to_outplace_tensor, args)
        kwargs = tree_map(to_outplace_tensor, kwargs)

        outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss: torch.Tensor):
        loss.backward()

        self.module.zero_grad(set_to_none=True)
        self.fetcher.clear()
        HookParam.release_fetcher()
