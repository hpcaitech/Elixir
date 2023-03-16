from collections import defaultdict
from copy import copy
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.utils._pytree import tree_map

from elixir import gpu_device
from elixir.chunk import Chunk, ChunkFetcher, ChunkGroup, MemoryPool, TensorState
from elixir.chunk.scheduler import FIFOScheduler, PrefetchScheduler
from elixir.hook import HookParam
from elixir.parameter import FakeTensor, OutplaceTensor
from elixir.search import SearchResult


def get_param_optim_data(param_data: torch.Tensor, param_dtype: torch.dtype):
    param_data = param_data.to(gpu_device())
    optim_data = param_data.clone() if param_data.dtype == torch.float else param_data.float()
    param_data = param_data.to(param_dtype)
    return param_data, optim_data


class ElixirModule(nn.Module):
    """Use this class to wrap your model when using Elixir. Don't know what should be written here.
    But some docstring is needed here.

    args:
        module: training module
        search_result: a SearchResult generated from a search algorithm in `elixir.search`
        process_group: the communication group, ussually dp parallel group
        prefetch: whether to use prefetch overlaping communication with computation
        dtype: the dtype used in training
    """

    def __init__(self,
                 module: nn.Module,
                 search_result: SearchResult,
                 process_group: ProcessGroup,
                 prefetch: bool = False,
                 dtype: torch.dtype = torch.float) -> None:
        super().__init__()

        assert dtype in {torch.float, torch.float16}

        self._set_module_outplace(module)
        self.module = module
        self.dtype = dtype
        self.use_amp = (dtype == torch.float16)
        self.process_group = process_group

        self.no_grad_state_dict = dict()
        self.grad_state_dict = dict()
        self.__init_chunk_group(search_result)
        self.__init_chunk_fetcher(search_result, prefetch)

        for name, param in module.named_parameters():
            if not param.requires_grad:
                assert name in self.no_grad_state_dict
                continue
            assert name in self.grad_state_dict
            param.register_hook(partial(self._gradient_handler, param=param))
            param.__class__ = HookParam

    def __init_chunk_group(self, sr: SearchResult):
        state_dict = self.module.state_dict(keep_vars=True)
        for name, tensor in state_dict.items():
            if isinstance(tensor, nn.Parameter):
                # deal with parameters
                if tensor.requires_grad:
                    self.grad_state_dict[name] = tensor
                else:
                    self.no_grad_state_dict[name] = tensor
                    # polish no-grad parameters
                    tensor.data = tensor.data.to(dtype=self.dtype, device=gpu_device())
            else:
                # deal with buffers
                tensor.data = tensor.data.to(dtype=self.dtype, device=gpu_device())

        empty_mp = MemoryPool('cuda')
        empty_mp.allocate()

        self.param_chunk_group = sr.chunk_group
        self.optim_chunk_group = ChunkGroup(empty_mp)
        self.param_to_optim = dict()
        vis_set = set()

        for plan in sr.param_chunk_plans:
            assert plan.chunk_dtype == self.dtype
            # optimizer chunks should not be gathered
            optim_kwargs = copy(plan.kwargs)
            if 'rcache_fused' in optim_kwargs:
                optim_kwargs['rcache_fused'] = False

            p_chunk = self.param_chunk_group.open_chunk(chunk_size=plan.chunk_size,
                                                        chunk_dtype=plan.chunk_dtype,
                                                        process_group=self.process_group,
                                                        chunk_config=plan.kwargs)
            o_chunk = self.optim_chunk_group.open_chunk(chunk_size=plan.chunk_size,
                                                        chunk_dtype=torch.float,
                                                        process_group=self.process_group,
                                                        chunk_config=optim_kwargs)

            for name in plan.name_list:
                param = self.grad_state_dict[name]
                # TODO(helson): deal with lazy init
                param_data, optim_data = get_param_optim_data(param.data, self.dtype)
                param.data = param_data
                p_chunk.append_tensor(param)
                o_chunk.append_tensor(optim_data)
                self.param_to_optim[param] = optim_data

                vis_set.add(param)

            self.param_chunk_group.close_chunk(p_chunk)
            self.optim_chunk_group.close_chunk(o_chunk)
            p_chunk.init_pair(o_chunk)

        # sanity check: every parameter needed gradient has been initialized
        for param in self.module.parameters():
            if param.requires_grad:
                assert param in vis_set

    def __init_chunk_fetcher(self, sr: SearchResult, prefetch: bool):
        scheduler = None
        if prefetch:
            scheduler = PrefetchScheduler(process_list=sr.param_called_per_step)
        else:
            scheduler = FIFOScheduler()

        self.fetcher = ChunkFetcher(scheduler, self.param_chunk_group, overlap=prefetch)

    def _gradient_handler(self, grad: torch.Tensor, param: nn.Parameter):
        # create an empty tensor
        empty_grad = torch.empty_like(grad)
        empty_grad.storage().resize_(0)

        with torch._C.DisableTorchFunction():
            chunk = self.fetcher.get_one_chunk(param)
            if chunk.tensors_info[param].state != TensorState.HOLD_AFTER_BWD:
                raise RuntimeError()
            self.fetcher.group.tensor_trans_state(param, TensorState.READY_FOR_REDUCE)
            chunk.copy_tensor_to_chunk_slice(param, grad)
            self.fetcher.reduce_chunk(chunk)

        return empty_grad

    def _lazy_init_check(self, m: nn.Module):
        # TODO(helson): deal with lazy init
        return False

    def _set_module_outplace(self, m: nn.Module):
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

    def state_dict(self, destination=None, prefix='', keep_vars=False, only_rank_0: bool = False):
        assert keep_vars is False, 'state_dict can not keep variables in ElixirModule'
        # make sure that the variables are kept, we shall detach them later
        module_state_dict = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=True)

        optim_to_names = defaultdict(list)
        for name, tensor in module_state_dict.items():
            if isinstance(tensor, nn.Parameter) and tensor.requires_grad:
                optim_data = self.param_to_optim.get(self.grad_state_dict[name])
                optim_to_names[optim_data].append(name)
            else:
                module_state_dict[name] = tensor.detach()

        def update_state_dict(chunks: Iterable[Chunk]):
            for c in chunks:
                for op, cp in zip(c.get_tensors(), c.get_cpu_copy(only_rank_0)):
                    for name in optim_to_names.get(op):
                        module_state_dict[name] = cp

        update_state_dict(self.optim_chunk_group.fused_chunks)
        update_state_dict(self.optim_chunk_group.float_chunks)

        return module_state_dict
