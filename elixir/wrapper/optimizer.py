import math
from enum import Enum
from typing import Any, Dict, Set, Tuple

import torch
import torch.distributed as dist
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.nn.optimizer import ColossalaiOptimizer, CPUAdam, FusedAdam, HybridAdam
from torch.nn import Parameter
from torch.optim import Optimizer

from elixir.chunk import Chunk, ChunkGroup
from elixir.utils import gpu_device

from .module import ElixirModule

_AVAIL_OPTIM_LIST = {FusedAdam, CPUAdam, HybridAdam}


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class ElixirOptimizer(ColossalaiOptimizer):

    def __init__(self,
                 module: ElixirModule,
                 optimizer: Optimizer,
                 initial_scale: float = 65536,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**24,
                 max_norm: float = 0.0,
                 norm_type: float = 2.0):

        super().__init__(optimizer)
        assert isinstance(module, ElixirModule)
        assert type(optimizer) in _AVAIL_OPTIM_LIST, 'You should use an optimizer in the available list:\n' \
            f'{_AVAIL_OPTIM_LIST}'

        self.module = module
        self.param_chunk_group = module.param_chunk_group
        self.optim_chunk_group = module.optim_chunk_group

        self.optim_state = OptimState.UNSCALED
        self.param_to_range: Dict[Parameter, Tuple[int, int]] = dict()
        self.param_to_optim_chunk: Dict[Parameter, Chunk] = dict()
        self.param_chunk_set: Set[Chunk] = self.param_chunk_group.fused_chunks.union(
            self.param_chunk_group.float_chunks)

        self.clipping_flag = max_norm > 0.0
        self.max_norm = max_norm
        if self.clipping_flag:
            assert norm_type == 2.0, 'ElixirOptimizer only supports L2 norm now'

        self.__init__optimizer()

        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=gpu_device())

    def _set_grad_ptr(self):
        for group in self.param_groups:
            for fake_param in group['params']:
                chunk32 = self.param_to_chunk32[fake_param]
                begin, end = self.param_to_range[fake_param]
                chunk16 = chunk32.paired_chunk

                fake_param.data = chunk16.payload[begin:end]
                fake_param.grad = fake_param.data
                fake_param.data = chunk32.payload[begin:end]

    def _update_fp16_params(self):
        none_tensor = torch.empty([0])
        for group in self.param_groups:
            for fake_param in group['params']:
                assert fake_param.grad is None
                fake_param.data = none_tensor.to(fake_param.device)

        for chunk16 in self.chunk16_set:
            chunk16.optim_update()

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        # all-reduce across global group
        dist.all_reduce(self._found_overflow)

        return self._found_overflow.item() > 0

    def _clear_global_norm(self) -> None:
        for c16 in self.chunk16_set:
            c16.l2_norm = None

    def _calc_global_norm(self) -> float:
        norm_sqr: float = 0.0
        group_to_norm = dict()
        for c16 in self.chunk16_set:
            assert c16.l2_norm is not None

            if c16.is_gathered:
                norm_sqr += c16.l2_norm
            else:
                # this chunk is sharded, use communication to collect total norm
                if c16.torch_pg not in group_to_norm:
                    group_to_norm[c16.torch_pg] = 0.0
                group_to_norm[c16.torch_pg] += c16.l2_norm

            c16.l2_norm = None    # clear l2 norm

        comm_buffer = torch.zeros(1, dtype=torch.float, device=get_current_device())
        for group, part_norm in group_to_norm.items():
            comm_buffer.fill_(part_norm)
            dist.all_reduce(comm_buffer, group=group)
            norm_sqr += comm_buffer.item()

        global_norm = math.sqrt(norm_sqr)
        return global_norm

    def _get_combined_scale(self):
        loss_scale = 1

        if self.optim_state == OptimState.SCALED:
            loss_scale = self.loss_scale
            self.optim_state = OptimState.UNSCALED

        combined_scale = loss_scale
        if self.clipping_flag:
            total_norm = self._calc_global_norm()
            clip = ((total_norm / loss_scale) + 1e-6) / self.max_norm
            if clip > 1:
                combined_scale = clip * loss_scale

        if combined_scale == 1:
            return -1
        else:
            return combined_scale

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        self._maybe_move_fp32_params()
        self._set_grad_ptr()

        found_inf = self._check_overflow()
        if found_inf:
            self.optim_state = OptimState.UNSCALED    # no need to unscale grad
            self.grad_scaler.update(found_inf)    # update gradient scaler
            self._logger.info(f'Found overflow. Skip step')
            self._clear_global_norm()    # clear recorded norm
            self.zero_grad()    # reset all gradients
            self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()
        self.grad_scaler.update(found_inf)

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self._register_states()
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float, norm_type: float = 2.0):
        raise NotImplementedError

    def backward(self, loss: torch.Tensor):
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.module.backward(loss)

    def backward_by_grad(self, tensor: torch.Tensor, grad: torch.Tensor):
        # This function is called except the last stage of pipeline parallel
        # It receives the scaled grad from the previous rank
        # No need to scale the grad again
        # Need to unscale when optimizing
        self.optim_state = OptimState.SCALED
        self.module.backward_by_grad(tensor, grad)

    def _register_states_(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                state = self.optim.state[p]
                for val in state.values():
                    if isinstance(val, torch.Tensor):
                        self.chunk_manager.add_extern_static_tensor(val)

    def __init__optimizer(self):

        def get_range_pair(local_chunk: Chunk, local_param: Parameter):
            param_info = local_chunk.tensors_info[local_param]
            if local_chunk.keep_gathered:
                return param_info.offset, param_info.end
            begin = max(0, param_info.offset - local_chunk.shard_begin)
            end = min(local_chunk.shard_size, param_info.end - local_chunk.shard_begin)
            return begin, end

        for group in self.optimizer.param_groups:
            fake_params_list = list()

            for param in group['params']:
                if not param.requires_grad:
                    continue

                param_chunk = self.module.fetcher.get_one_chunk(param)
                range_pair = get_range_pair(chunk16, param)
                if range_pair[0] >= range_pair[1]:
                    continue

                grad_device = self.module.grads_device[param]
                fake_param = torch.nn.Parameter(torch.empty([0], device=grad_device))
                self.param_to_chunk32[fake_param] = chunk16.paired_chunk
                self.param_to_range[fake_param] = range_pair

                fake_params_list.append(fake_param)

            group['params'] = fake_params_list
