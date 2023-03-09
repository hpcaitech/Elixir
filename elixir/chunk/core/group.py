from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .chunk import Chunk
from .memory_pool import MemoryPool, TensorBlock
from .states import TensorState


class ChunkGroup(object):

    def __init__(self, rcache: MemoryPool) -> None:
        super().__init__()
        self.rcache = rcache
        self.fused_chunks: set[Chunk] = set()
        self.float_chunks: set[Chunk] = set()
        self.ten_to_chunk: Dict[torch.Tensor, Chunk] = dict()

        self.accessed_fused_chunks: set[Chunk] = set()
        self.accessed_float_chunks: set[Chunk] = set()

    def __add_to_accset(self, chunk: Chunk):
        if chunk.rcache_fused:
            self.accessed_fused_chunks.add(chunk)
        else:
            self.accessed_float_chunks.add(chunk)

    def __remove_from_accset(self, chunk: Chunk):
        if chunk.rcache_fused:
            self.accessed_fused_chunks.remove(chunk)
        else:
            self.accessed_float_chunks.remove(chunk)

    def inside_check(self, chunk: Chunk) -> None:
        # check whether the chunk is in this ChunkGroup
        if chunk.rcache_fused:
            assert chunk in self.fused_chunks
        else:
            assert chunk in self.float_chunks

    def is_accessed(self, chunk: Chunk) -> bool:
        # sanity check
        self.inside_check(chunk)

        if chunk.rcache_fused:
            return (chunk in self.accessed_fused_chunks)
        else:
            return (chunk in self.accessed_float_chunks)

    def allocate_chunk(self,
                       tensor_list: list[torch.Tensor],
                       chunk_size: int,
                       chunk_dtype: torch.dtype,
                       process_group: ProcessGroup,
                       chunk_config: Optional[Dict] = None) -> Chunk:

        if chunk_config is None:
            chunk_config = {}

        new_chunk = Chunk(rcache=self.rcache,
                          chunk_size=chunk_size,
                          chunk_dtype=chunk_dtype,
                          process_group=process_group,
                          **chunk_config)
        # sanity check
        if not new_chunk.rcache_fused:
            assert new_chunk.chunk_size == self.rcache.public_block_size
            assert new_chunk.chunk_dtype == self.rcache.public_dtype
        # append tensors
        for t in tensor_list:
            new_chunk.append_tensor(t)
        new_chunk.close_chunk()
        # add the new chunk to the set of allocated chunks
        if new_chunk.rcache_fused:
            self.fused_chunks.add(new_chunk)
        else:
            self.float_chunks.add(new_chunk)
        # add the new chunk to the mapping
        for t in tensor_list:
            assert t not in self.ten_to_chunk
            self.ten_to_chunk[t] = new_chunk

        return new_chunk

    def tensors_to_chunks(self, tensor_list: list[torch.Tensor]) -> list[Chunk]:
        chunk_list = list()
        for tensor in tensor_list:
            chunk = self.ten_to_chunk.get(tensor)
            if chunk not in chunk_list:
                chunk_list.append(chunk)
        chunk_list.sort(key=lambda c: c.chunk_id)
        return chunk_list

    def rcache_enough_check(self, chunk: Chunk) -> bool:
        if chunk.rcache_fused:
            return True
        # the flag of free block list
        flag = bool(self.rcache.public_free_blocks)
        return flag

    def access_chunk(self, chunk: Chunk) -> bool:
        self.inside_check(chunk)
        # if this chunk is accessed already, return False
        if self.is_accessed(chunk):
            return False

        if chunk.rcache_fused:
            block = None
        else:
            block = self.rcache.get_public_block()
        chunk.access_chunk(block)
        self.__add_to_accset(chunk)
        return True

    def release_chunk(self, chunk: Chunk) -> bool:
        self.inside_check(chunk)
        assert self.is_accessed(chunk)
        assert chunk.scatter_check

        block = chunk.release_chunk()
        if block:
            self.rcache.free_public_block(block)
        self.__remove_from_accset(chunk)
        return True

    def reduce_chunk(self, chunk: Chunk) -> bool:
        self.inside_check(chunk)
        assert self.is_accessed(chunk)
        assert chunk.reduce_check

        block = chunk.reduce_chunk()
        if block:
            self.rcache.free_public_block(block)
        self.__remove_from_accset(chunk)
        return True

    def tensor_trans_state(self, tensor: torch.Tensor, state: TensorState):
        chunk = self.ten_to_chunk.get(tensor)
        chunk.tensor_trans_state(tensor, state)
