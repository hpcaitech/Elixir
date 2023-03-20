from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkGroup, MemoryPool
from elixir.tracer.utils import meta_copy

from .result import ChunkPlan
from .utils import to_meta_tensor


class SearchBase(ABC):
    """A basic class for search algorithms.
    """

    def __init__(self,
                 module: nn.Module,
                 dtype: torch.dtype = torch.float,
                 prefetch: bool = False,
                 verbose: bool = False,
                 inp=None,
                 step_fn=None) -> None:

        self.unified_dtype = dtype
        self.meta_module = meta_copy(module, partial(to_meta_tensor, dtype=self.unified_dtype))
        self.prefetch_flag = prefetch
        self.verbose = verbose
        self.param_to_name = {param: name for name, param in self.meta_module.named_parameters()}

        self.public_block_size = 1024
        self.public_block_number = 0

        if self.prefetch_flag:
            assert inp is not None and step_fn is not None
            # TODO(helson): add parameter order tracing

    @abstractmethod
    def private_truncate(param: nn.Parameter) -> int:
        """A function used to truncate the length of a private chunk,
        which only contains one parameter.
        """
        pass

    @abstractmethod
    def public_trucate(length: int) -> int:
        """A function used to trucate the length of all publick chunks
        """
        pass

    @abstractmethod
    def search(*args, **kwargs) -> Tuple:
        """The core search function. It returns a list of chunk plans.
        """
        pass

    def generate_chunk_plans(self, private_group, publick_groups) -> list[ChunkPlan]:
        plans = list()
        for param in private_group:
            chunk_size = self.private_truncate(param)
            chunk_dtype = param.dtype
            chunk_kwargs = dict(rcache_fused=True)
            chunk_plan = ChunkPlan(name_list=[self.param_to_name[param]],
                                   chunk_size=chunk_size,
                                   chunk_dtype=chunk_dtype,
                                   kwargs=chunk_kwargs)
            plans.append(chunk_plan)

        self.public_block_size = self.public_trucate(self.public_block_size)
        public_chunk_size = self.public_block_size
        public_chunk_dtype = self.unified_dtype
        for group in publick_groups:
            chunk_kwargs = {}
            chunk_plan = ChunkPlan(name_list=[self.param_to_name[p] for p in group],
                                   chunk_size=public_chunk_size,
                                   chunk_dtype=public_chunk_dtype,
                                   kwargs=chunk_kwargs)
            plans.append(chunk_plan)

        return plans

    def allocate_chunk_group(self, chunk_plans: list[ChunkPlan]) -> ChunkGroup:
        block_require_list = list()
        for plan in chunk_plans:
            kwargs = plan.kwargs
            if kwargs.get('rcache_fused', False):
                block_require_list.append(BlockRequire(plan.chunk_size, plan.chunk_dtype))

        mp = MemoryPool('cuda')
        mp.allocate(public_dtype=self.unified_dtype,
                    public_block_size=self.public_block_size,
                    public_block_number=self.public_block_number,
                    private_block_list=block_require_list)
        return ChunkGroup(mp)
