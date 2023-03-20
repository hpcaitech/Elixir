from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from elixir.chunk import BlockRequire, ChunkGroup, MemoryPool
from elixir.tracer.param_tracer import generate_tf_order
from elixir.tracer.utils import meta_copy
from elixir.utils import print_rank_0

from .result import ChunkPlan
from .utils import to_meta_tensor


class SearchBase(ABC):
    """A basic class for search algorithms.

    args:
        module: the model to be searched
        dtype: the unified dtype of all parameters
        prefetch: whether to prefetch chunks during training
        verbose: whether to print search details
        inp: a dictionary, the example input of the model
        step_fn: the example step function of the model
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

        self.param_per_step = None
        if self.prefetch_flag:
            assert inp is not None and step_fn is not None
            self.param_per_step = generate_tf_order(self.meta_module, inp, step_fn, dtype)
            if self.verbose:
                print_rank_0('Prefetch enabled: the called order of parameters')
                for i, step in enumerate(self.param_per_step):
                    print_rank_0(f'step {i}: {step}')

    @abstractmethod
    def private_truncate(self, param: nn.Parameter) -> int:
        """A function used to truncate the length of a private chunk,
        which only contains one parameter.
        """
        pass

    @abstractmethod
    def public_trucate(self, length: int) -> int:
        """A function used to trucate the length of all publick chunks
        """
        pass

    @abstractmethod
    def search(self, *args, **kwargs) -> Tuple:
        """The core search function. It returns a tuple of a private group and public groups.
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

        if self.verbose:
            print_rank_0(f'Chunk plans: total {len(plans)} chunks')
            for i, plan in enumerate(plans):
                print_rank_0(f'plan {i}: {plan}')

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

        if self.verbose:
            print_rank_0(
                f'Memory pool (rcache): {mp}\n\tblock size -> {mp.public_block_size}, block number -> {mp.public_free_cnt}'
            )

        return ChunkGroup(mp)
