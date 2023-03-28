import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd.profiler_util import _format_memory

from elixir.tracer.memory_tracer import cuda_memory_profiling
from elixir.utils import gpu_device, print_rank_0

from .base import SearchBase
from .result import SearchResult
from .simulator import find_optimal_chunk_size
from .utils import find_search_range, get_multi_used_params, to_divide

dtype_to_es = {torch.float16: 2, torch.float32: 4, torch.float64: 8}


class SearchOptimal(SearchBase):

    def __init__(self,
                 module: nn.Module,
                 default_group_size: int,
                 activation_fragment_factor: float = 1.3,
                 allocation_fragment_factor: float = 0.95,
                 dtype: torch.dtype = torch.float,
                 verbose: bool = False,
                 overlap: bool = False,
                 inp=None,
                 step_fn=None) -> None:
        # as for optimal search, we must profile the model first
        super().__init__(module, dtype, True, verbose, inp, step_fn)
        # profile cuda memory usage
        memo_usage = cuda_memory_profiling(model=self.meta_module, inp=inp, step_fn=step_fn, dtype=dtype)
        # get the maximum memory usage of activation
        predict_activation = memo_usage['activation_occ']
        # calculate the total capacity of the current device
        gpu_memory = torch.cuda.get_device_properties(gpu_device()).total_memory
        # allowed capacity = allocation_fragment_factor * (total capacity - activation_fragment_factor * activation)
        self.cuda_capacity = int(allocation_fragment_factor *
                                 (gpu_memory - activation_fragment_factor * predict_activation))
        self.cuda_elements = self.cuda_capacity // dtype_to_es.get(dtype)

        if self.cuda_capacity < 0:
            raise RuntimeError('optimal search: activation is too large, please reduce batch size')

        if self.verbose:
            print_rank_0('Predict memory usage:')
            for k, v in memo_usage.items():
                print_rank_0(f'{k}: {_format_memory(v)}')
            print_rank_0(f'allowed allocation space: {_format_memory(self.cuda_capacity)}')
            print_rank_0(f'allowed {dtype} elements: {self.cuda_elements}')

        self.default_group_size = default_group_size
        self.comm_overlap = overlap

    def private_truncate(self, param: nn.Parameter) -> int:
        return to_divide(param.numel(), self.default_group_size)

    def public_trucate(self, length: int) -> int:
        return to_divide(length, self.default_group_size)

    def search(self) -> Tuple:
        min_chunk_size, max_chunk_size, search_interval = find_search_range(self.meta_module)
        # get multi-used parameters
        private_params = get_multi_used_params(self.meta_module)
        # subtract the footprint of fused parameters
        for param in private_params:
            self.cuda_elements -= param.numel()
        if self.cuda_elements < 0:
            raise RuntimeError('optimal search: no enough space for fused parameters')

        # initialize public params in the called order
        public_params = list()
        public_param_set = set()
        name_to_param = {name: param for name, param in self.meta_module.named_parameters()}
        for name_set in self.param_per_step:
            for name in name_set:
                param = name_to_param.get(name)
                if param in private_params or param in public_param_set:
                    continue
                public_params.append(param)
                public_param_set.add(param)
        del name_to_param
        del public_param_set

        # collect the number of elements of each parameter
        public_numels = [p.numel() for p in public_params]
        # calculate the sumary of all parameters
        total_size = sum(public_numels)
        # collect the name for each public parameters
        public_param_names = [self.param_to_name[p] for p in public_params]

        if total_size <= min_chunk_size:
            public_block_size = total_size
            n_blocks = 1
            waste_size = 0
        else:
            public_block_size, n_blocks, waste_size = find_optimal_chunk_size(
            # pre-commit: do not rearrange
                param_per_step=self.param_per_step,
                param_names=public_param_names,
                param_numels=public_numels,
                cuda_elements=self.cuda_elements,
                overlap=self.comm_overlap,
                min_range=min_chunk_size,
                max_range=max_chunk_size,
                interval=search_interval)
        # truncate the size of public blocks
        public_block_size = self.public_trucate(public_block_size)
        # subtract the space of blocks
        self.cuda_elements -= n_blocks * public_block_size
        if self.cuda_elements < 0:
            raise RuntimeError('no enough space for unfused parameters')

        if self.verbose:
            if total_size == 0:
                waste_percentage = 0
            else:
                waste_percentage = 100 * waste_size / total_size
            print_rank_0(
                f'Optimal search result: chunk size = {public_block_size}, waste percentage = {waste_percentage: .1f} %'
            )

        # initialize the mapping from parameters to chunks
        param_to_chunk_id = dict()
        chunk_id = 0
        # deal with private parameters
        for p in private_params:
            param_to_chunk_id[p] = chunk_id
            chunk_id += 1
        # record the upper bound
        private_id_upperbound = chunk_id
        # deal with public parameters
        last_left = 0
        for p in public_params:
            p_size = p.numel()

            if last_left < p_size:
                last_left = public_block_size
                chunk_id += 1

            assert last_left >= p_size

            last_left -= p_size
            param_to_chunk_id[p] = chunk_id

        # initailize public groups
        public_number_chunks = chunk_id - private_id_upperbound
        public_groups = [[] for _ in range(public_number_chunks)]
        for p in public_params:
            public_chunk_id = param_to_chunk_id[p] - private_id_upperbound - 1
            public_groups[public_chunk_id].append(p)

        if total_size == 0:
            n_blocks = 0

        self.public_block_size = public_block_size
        self.public_block_number = n_blocks

        return (private_params, public_groups)


def optimal_search(
    # pre-commit: do not rearrange
        m: nn.Module,
        group_size: int,
        unified_dtype: torch.dtype = torch.float,
        optimizer_type: str = 'Adam',
        overlap: bool = False,
        verbose: bool = False,
        inp=None,
        step_fn=None) -> SearchResult:

    search_class = SearchOptimal(
    # pre-commit: do not rearrange
        module=m,
        default_group_size=group_size,
        dtype=unified_dtype,
        verbose=verbose,
        overlap=overlap,
        inp=inp,
        step_fn=step_fn)

    private_group, public_groups = search_class.search()
    chunk_plans = search_class.generate_chunk_plans(private_group, public_groups)

    if unified_dtype == torch.float16:
        master_weight_factor = 2
    elif unified_dtype == torch.float:
        master_weight_factor = 1
    else:
        raise NotImplementedError

    if optimizer_type == 'SGD':
        extra_sotre_factor = 1
    elif optimizer_type == 'Adam':
        extra_sotre_factor = 2
    else:
        raise NotImplementedError
    os_factor = 1 + (1 + extra_sotre_factor) * master_weight_factor

    for (i, plan) in enumerate(chunk_plans):
        param_os_size = os_factor * plan.chunk_size // group_size
        if search_class.cuda_elements >= param_os_size:
            plan.kwargs['shard_device'] = gpu_device()
            search_class.cuda_elements -= param_os_size
        else:
            plan.kwargs['shard_device'] = torch.device('cpu')
            plan.kwargs['cpu_pin_memory'] = True

        print_rank_0(f"chunk {i}: shard device -> {plan.kwargs['shard_device']}")

    chunk_group = search_class.allocate_chunk_group(chunk_plans)

    return SearchResult(chunk_group=chunk_group,
                        chunk_plans=chunk_plans,
                        param_called_per_step=search_class.param_per_step)
