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
        self.main_stream = torch.cuda.current_stream()
        self.fetching_chunk = None
        self.prefetch_stream = torch.cuda.Stream()
        self.reducing_flag = False
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
        if self.fetching_chunk is not None:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            self.fetching_chunk = None

    def wait_reduce(self):
        if self.reducing_flag:
            torch.cuda.current_stream().wait_stream(self.reduce_stream)
            self.reducing_flag = False

    def wait_main(self):
        torch.cuda.current_stream().wait_stream(self.main_stream)

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

        prefetch_hit = False
        # wait async prefetch
        if self.fetching_chunk is not None and self.fetching_chunk in chunks:
            self.wait_prefetch()
            prefetch_hit = True
        # filter accessed chunks
        scattered = self.filter_chunks(chunks)
        # sanity check: upload should wait for prefetch
        if self.fetching_chunk is not None:
            assert len(scattered) == 0
        # all chunks are on the rcache
        if len(scattered) == 0:
            # prefetch if there is a hit above
            if prefetch_hit:
                self.prefetch(chunks)
            return

        # wait async reduce
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

        if self.overlap_flag:
            assert self.fetching_chunk is None
            self.prefetch(chunks)

    def reduce_chunk(self, chunk: Chunk):
        if not chunk.reduce_check:
            return False

        if self.overlap_flag:
            context = torch.cuda.stream
            self.reducing_flag = True
        else:
            context = nullcontext

        self.scheduler.remove(chunk)
        with context(self.reduce_stream):
            if self.overlap_flag:
                self.wait_main()
            self.group.reduce_chunk(chunk)

    def prefetch(self, chunks: list[Chunk]):
        # TODO: this instruction
        next_chunk = self.scheduler.get_next_chunk(chunks)
        # return if there is no next scattered chunk
        if next_chunk is None or self.group.is_accessed(next_chunk):
            return

        evict_chunk = None
        if not self.group.rcache_enough_check(next_chunk):
            maybe_chunk = self.scheduler.top()
            # if there is no chunk can be evicted, just return
            if maybe_chunk is None:
                return
            # otherwise, release this chunk
            self.scheduler.remove(maybe_chunk)
            evict_chunk = maybe_chunk

        with torch.cuda.stream(self.prefetch_stream):
            if self.reducing_flag:
                self.wait_reduce()
            self.wait_main()
            self.fetching_chunk = next_chunk
            if evict_chunk is not None:
                self.group.release_chunk(evict_chunk)
            self.group.access_chunk(next_chunk)

    def step(self):
        self.scheduler.step()
        self.current_step += 1
