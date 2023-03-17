from contextlib import nullcontext

import torch

from .core import Chunk, ChunkGroup, TensorState
from .scheduler import ChunkScheduler


class ChunkFetcher(object):

    def __init__(self, scheduler: ChunkScheduler, group: ChunkGroup, overlap: bool = False) -> None:
        self.scheduler: ChunkScheduler = scheduler
        self.group: ChunkGroup = group
        self.current_step = -1

        self.overlap_flag = overlap
        self.fetching_chunk = None
        self.prefetch_stream = torch.cuda.Stream()
        self.reducing_chunk = None
        self.reduce_stream = torch.cuda.Stream()

    def reset(self):
        self.scheduler.reset()
        self.current_step = -1

    def clear(self):
        self.scheduler.clear()

    def trans_to_compute(self, tensors: list[torch.Tensor]):
        # update tensor states
        for t in tensors:
            self.group.tensor_trans_state(t, TensorState.COMPUTE)
        # chunk operations
        chunks = self.group.tensors_to_chunks(tensors)
        for chunk in chunks:
            self.scheduler.remove(chunk)
        return chunks

    def trans_to_hold(self, tensors: list[torch.Tensor], phase: str):
        assert phase in ('f', 'b')
        next_state = TensorState.HOLD if phase == 'f' else TensorState.HOLD_AFTER_BWD
        # update tensor states
        for t in tensors:
            self.group.tensor_trans_state(t, next_state)
        # chunk operations
        chunks = self.group.tensors_to_chunks(tensors)
        for chunk in chunks:
            if chunk.scatter_check:
                self.scheduler.add(chunk)

    def wait_prefetch(self):
        assert self.fetching_chunk is not None
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        self.fetching_chunk = None

    def wait_reduce(self):
        assert self.reducing_chunk is not None
        torch.cuda.current_stream().wait_stream(self.reduce_stream)
        self.reducing_chunk = None

    def get_one_chunk(self, tensor: torch.Tensor) -> Chunk:
        return self.group.ten_to_chunk.get(tensor)

    def get_chunks(self, tensors: list[torch.Tensor]) -> list[Chunk]:
        return self.group.tensors_to_chunks(tensors)

    def is_in_fused(self, tensor: torch.Tensor):
        chunk = self.get_one_chunk(tensor)
        return chunk.rcache_fused

    def filter_chunks(self, chunks: list[Chunk]):
        return list(filter(lambda c: not self.group.is_accessed(c), chunks))

    def fetch_chunks(self, chunks: list[Chunk]):
        # make step + 1
        self.step()
        # wait async prefetch
        if self.fetching_chunk is not None and self.fetching_chunk in chunks:
            self.wait_prefetch()
        scattered = self.filter_chunks(chunks)

        # sanity check: upload should wait for prefetch
        if self.fetching_chunk:
            assert len(scattered) == 0
        # all chunks are on the rcache
        if len(scattered) == 0:
            return

        # wait async reduce
        if self.reducing_chunk is not None:
            self.wait_reduce()

        for chunk in scattered:
            # if the rcache is not enough, just release a chunk
            if not self.group.rcache_enough_check(chunk):
                maybe_chunk = self.scheduler.top()
                # print(f'Evicting {chunk.chunk_id} -> {maybe_chunk.chunk_id}')
                if maybe_chunk is None:
                    raise RuntimeError('R cache is not enough. Try to allocate more.')
                self.scheduler.remove(maybe_chunk)
                self.group.release_chunk(maybe_chunk)

            # print('Accessing', chunk.chunk_id)
            self.group.access_chunk(chunk)

    def reduce_chunk(self, chunk: Chunk) -> bool:
        if not chunk.reduce_check:
            return False

        if self.overlap_flag:
            context = torch.cuda.stream
            self.reducing_chunk = chunk
        else:
            context = nullcontext

        self.scheduler.remove(chunk)
        with context(self.reduce_stream):
            self.group.reduce_chunk(chunk)

        return True

    def prefetch(self, chunks: list[Chunk]):
        # TODO: this instruction
        next_chunk = self.scheduler.get_next_chunk(chunks)
        # return if there is no next scattered chunk
        if next_chunk is None or self.group.is_accessed(next_chunk):
            return

        if not self.group.rcache_enough_check(next_chunk):
            maybe_chunk = self.scheduler.top()
            # if there is no chunk can be evicted, just return
            if maybe_chunk is None:
                return
            # otherwise, release this chunk
            self.scheduler.remove(maybe_chunk)
            self.group.release_chunk(maybe_chunk)

        self.fetching_chunk = next_chunk
        with torch.cuda.stream(self.prefetch_stream):
            self.group.access_chunk(next_chunk)

    def step(self):
        self.scheduler.step()
        self.current_step += 1
