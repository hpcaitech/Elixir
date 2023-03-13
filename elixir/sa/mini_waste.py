import math
from functools import partial

import torch
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkGroup, MemoryPool
from elixir.tracer.utils import meta_copy

from .result import SearchResult
from .utils import find_minimum_waste_size, get_multi_used_params, to_divide, to_meta_tensor

dtype_to_es = {torch.float16: 2, torch.float32: 4, torch.float64: 8}


def is_leaf_module(m: torch.nn.Module) -> bool:
    return ((m.__module__.startswith('torch.nn') or m.__module__.startswith('torch.ao.nn'))
            and not isinstance(m, torch.nn.Sequential))


def minimum_waste_sa(m: nn.Module,
                     group_size: int,
                     min_chunk_occ_mb: float = 32,
                     max_chunk_occ_mb: float = 96,
                     test_interval: int = 1024,
                     unified_dtype: torch.dtype = torch.float) -> SearchResult:

    # transform unit first
    element_size = dtype_to_es.get(unified_dtype)
    min_chunk_size = math.ceil(min_chunk_occ_mb * 1024**2) // element_size
    max_chunk_size = math.ceil(max_chunk_occ_mb * 1024**2) // element_size
    # get a meta copy of the model
    m = meta_copy(m, partial(to_meta_tensor, dtype=unified_dtype))

    param_to_name = {param: name for name, param in m.named_parameters()}
    private_params = get_multi_used_params(m)
    public_params = [p for p in m.parameters() if p not in private_params]
    public_numels = [p.numel() for p in public_params]
    assert len(public_params) > 0

    total_size = sum(public_numels)
    if total_size <= min_chunk_size:
        public_block_size = total_size
        waste_size = 0
    else:
        public_block_size, waste_size = find_minimum_waste_size(numel_group_list=[public_numels],
                                                                min_range=min_chunk_size,
                                                                max_range=max_chunk_size,
                                                                interval=test_interval)
    print(
        f'minimum waste searching ends: chunk size = {public_block_size}, waste percentage = {100 * waste_size / total_size: .1f} %'
    )

    # initialize the mapping from parameters to chunks
    param_to_chunk_id = dict()
    chunk_id = 0
    # deal with private parameters
    for p in private_params:
        param_to_chunk_id[p] = chunk_id
        chunk_id += 1
    # deal with public parameters
    last_left = public_block_size
    for p in public_params:
        if last_left >= p.numel():
            param_to_chunk_id[p] = chunk_id
            last_left -= p.numel()
        else:
            last_left = public_block_size
            chunk_id += 1

    # calculate the size of R cache
    max_lived_chunks = 0
    for module in m.modules():
        if is_leaf_module(module):
            param_set = set()
            for p in module.parameters():
                param_set.add(param_to_chunk_id[p])
            max_lived_chunks = max(max_lived_chunks, len(param_set))

    # initialized public groups
    public_groups = list()
    while public_params:
        p_list = list()
        chunk_id = param_to_chunk_id[public_params[0]]
        while public_params:
            p = public_params[0]
            if param_to_chunk_id[p] != chunk_id:
                break
            p_list.append(public_params.pop(0))
        public_groups.append(p_list)

    # get chunk configuration
    config_list = list()
    br_list = list()
    for p in private_params:
        block_size = to_divide(p.numel(), group_size)
        block_dtype = p.dtype
        br_list.append(BlockRequire(block_size, block_dtype))

        fused_config = dict(rcache_fused=True)
        init_dict = dict(name_list=[param_to_name[p]],
                         chunk_size=block_size,
                         chunk_dtype=block_dtype,
                         kwargs=fused_config)
        config_list.append(init_dict)

    public_block_size = to_divide(public_block_size, group_size)
    for p_list in public_groups:
        name_list = [param_to_name[p] for p in p_list]
        init_dict = dict(name_list=name_list, chunk_size=public_block_size, chunk_dtype=unified_dtype, kwargs=None)
        config_list.append(init_dict)

    # allocate a memory pool
    mp = MemoryPool('cuda')
    mp.allocate(public_dtype=unified_dtype,
                public_block_size=public_block_size,
                public_block_number=max_lived_chunks,
                private_block_list=br_list)
    chunk_group = ChunkGroup(mp)

    return SearchResult(chunk_group, config_list)
