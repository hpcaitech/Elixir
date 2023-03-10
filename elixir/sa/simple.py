import torch
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkGroup, MemoryPool
from elixir.tracer.utils import meta_copy

from .result import SearchResult
from .utils import get_multi_used_params, to_divide


def simple_sa(m: nn.Module,
              group_size: int,
              split_number: int = 10,
              test_dtype: torch.dtype = torch.float) -> SearchResult:

    def tensor_trans(t: torch.Tensor):
        # to meta
        meta_t = t.data.to(device='meta')
        # to the pointed dtype
        if t.is_floating_point():
            meta_t = meta_t.to(dtype=test_dtype)
        # pack it if t is a parameter
        # we should filter parameters with no grad
        if isinstance(t, nn.Parameter) and t.requires_grad:
            meta_t = nn.Parameter(meta_t)
        return meta_t

    m = meta_copy(m, tensor_trans)

    param_to_name = dict()
    for name, param in m.named_parameters():
        param_to_name[param] = name

    private_params = get_multi_used_params(m)
    public_params = list()
    public_groups = list()

    for param in m.parameters():
        if param not in private_params:
            public_params.append(param)

    len_public = len(public_params)
    split_number = min(len_public, split_number)
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
    del public_params

    # calculate the maximum summarized size
    max_sum_size = 0
    for p_list in public_groups:
        sum_size = sum([p.numel() for p in p_list])
        max_sum_size = max(max_sum_size, sum_size)
    max_sum_size = to_divide(max_sum_size, group_size)

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

    for p_list in public_groups:
        name_list = [param_to_name[p] for p in p_list]
        init_dict = dict(name_list=name_list, chunk_size=max_sum_size, chunk_dtype=test_dtype, kwargs=None)
        config_list.append(init_dict)

    # allocate a memory pool
    mp = MemoryPool('cuda')
    mp.allocate(public_dtype=test_dtype,
                public_block_size=max_sum_size,
                public_block_number=split_number // 2,
                private_block_list=br_list)
    chunk_group = ChunkGroup(mp)

    return SearchResult(chunk_group, config_list)
