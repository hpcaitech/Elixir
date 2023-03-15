import math
from functools import partial

import torch
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkGroup, MemoryPool
from elixir.tracer.utils import meta_copy

from .result import ChunkPlan, SearchResult
from .utils import get_multi_used_params, to_divide, to_meta_tensor


def simple_search(
    m: nn.Module,
    group_size: int,
    split_number: int = 10,
    allocate_factor: float = 0.6,
    unified_dtype: torch.dtype = torch.float,
    shard_device: torch.device = torch.device('cpu')) -> SearchResult:

    # get a meta copy of the model
    m = meta_copy(m, partial(to_meta_tensor, dtype=unified_dtype))

    # create a mapping from parameter to name
    param_to_name = {param: name for name, param in m.named_parameters()}
    # get multi-used parameters
    private_params = get_multi_used_params(m)
    # get parameters used only one time
    public_params = [p for p in m.parameters() if p not in private_params]

    # calculate the size of each group
    len_public = len(public_params)
    split_number = min(len_public, split_number)
    # allocate a list for groups
    public_groups = list()
    if split_number > 0:
        average_size = len_public // split_number
        left_size = len_public % split_number

        # set the size of each segment
        pack_size_list = [average_size] * split_number
        for i in range(split_number):
            if left_size > 0:
                pack_size_list[i] += 1
            left_size -= 1

        # split public parameters
        for i in range(split_number):
            p_list = list()
            for _ in range(pack_size_list[i]):
                p = public_params.pop(0)
                p_list.append(p)
            public_groups.append(p_list)
        assert len(public_params) == 0

        # calculate the maximum summarized size
        max_sum_size = 0
        for p_list in public_groups:
            sum_size = sum([p.numel() for p in p_list])
            max_sum_size = max(max_sum_size, sum_size)
        max_sum_size = to_divide(max_sum_size, group_size)
    else:
        max_sum_size = 0

    # get chunk configuration
    br_list = list()
    plan_list = list()

    for p in private_params:
        block_size = to_divide(p.numel(), group_size)
        block_dtype = p.dtype
        br_list.append(BlockRequire(block_size, block_dtype))

        chunk_kwargs = dict(rcache_fused=True, shard_device=shard_device)
        chunk_plan = ChunkPlan(name_list=[param_to_name[p]],
                               chunk_size=block_size,
                               chunk_dtype=block_dtype,
                               kwargs=chunk_kwargs)
        plan_list.append(chunk_plan)

    for p_list in public_groups:
        name_list = [param_to_name[p] for p in p_list]
        chunk_kwargs = dict(shard_device=shard_device)
        chunk_plan = ChunkPlan(name_list=name_list,
                               chunk_size=max_sum_size,
                               chunk_dtype=unified_dtype,
                               kwargs=chunk_kwargs)
        plan_list.append(chunk_plan)

    # allocate a memory pool
    mp = MemoryPool('cuda')
    mp.allocate(public_dtype=unified_dtype,
                public_block_size=max_sum_size,
                public_block_number=math.ceil(split_number * allocate_factor),
                private_block_list=br_list)
    chunk_group = ChunkGroup(mp)

    return SearchResult(chunk_group, plan_list)
