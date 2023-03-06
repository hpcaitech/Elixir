from contextlib import nullcontext

import torch

from .chunk import Chunk
from .group import ChunkGroup
from .scheduler import ChunkScheduler


class ChunkFetcher(object):

    def __init__(self, scheduler: ChunkScheduler, overlap: bool = False) -> None:
        self.scheduler: ChunkScheduler = scheduler
        self.group: ChunkGroup = scheduler.group
        self.current_step = 0

        self.overlap_flag = overlap
        self.fetching_chunk = None
        self.prefetch_stream = torch.cuda.Stream()
        self.reducing_chunk = None
        self.reduce_stream = torch.cuda.Stream()

    def reset(self):
        self.scheduler.reset()
        self.current_step = 0

    def clear(self):
        self.scheduler.clear()

    def to_compute(self, chunks: list[Chunk]):
        for chunk in chunks:
            self.scheduler.remove(chunk)

    def to_hold(self, chunks: list[Chunk]):
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

    def get_chunks(self, tensors: list[torch.Tensor]) -> list[Chunk]:
        return self.group.tensors_to_chunks(tensors)

    def filter_chunks(self, chunks: list[Chunk]):
        return filter(lambda c: not self.group.is_accessed(c), chunks)

    def fetch_chunks(self, chunks: list[Chunk]):
        # wait async prefetch
        if self.fetching_chunk is not None and self.fetching_chunk in chunks:
            self.wait_prefetch()
        scattered = self.filter_chunks(chunks)
        # all chunks are on the rcache
        if len(scattered) == 0:
            return
        # wait async reduce
        if self.reducing_chunk is not None:
            self.wait_reduce()

        for chunk in chunks:
            # if the rcache is not enough, just release a chunk
            if not self.group.rcache_enough_check(chunk):
                maybe_chunk = self.scheduler.top(step=self.current_step)
                if maybe_chunk is None:
                    raise RuntimeError('R cache is not enough. Try to allocate more.')
                self.scheduler.remove(maybe_chunk)
                self.group.release_chunk(maybe_chunk)

            self.group.access_chunk(chunk)

    def reduce_chunk(self, chunk: Chunk):
        if self.overlap_flag:
            context = torch.cuda.stream
            self.reducing_chunk = chunk
        else:
            context = nullcontext

        self.scheduler.remove(chunk)
        with context(self.reduce_stream):
            self.group.reduce_chunk(chunk)

    def prefetch(self, chunks: list[Chunk]):
        # TODO: this instruction
        next_chunk = self.scheduler.get_next_chunk(chunks)
        # return if there is no next scattered chunk
        if next_chunk is None or self.group.is_accessed(next_chunk):
            return

        if not self.group.rcache_enough_check(next_chunk):
            maybe_chunk = self.scheduler.top(step=self.current_step)
            # if there is no chunk can be evicted, just return
            if maybe_chunk is None:
                return
            # otherwise, release this chunk
            self.scheduler.remove(maybe_chunk)
            self.group.release_chunk(maybe_chunk)

        self.fetching_chunk = next_chunk
        with torch.cuda.stream(self.prefetch_stream):
            self.group.access_chunk(next_chunk)
