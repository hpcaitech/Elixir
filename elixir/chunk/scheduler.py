from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from .chunk import Chunk
from .group import ChunkGroup
from .memory_pool import MemoryPool, TensorBlock


class ChunkScheduler(ABC):

    def __init__(self, rcache: MemoryPool, group: ChunkGroup) -> None:
        super().__init__()
        self.rcache = rcache
        self.group = group

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def upload_chunk(self, chunk: Chunk, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def offload_chunk(self, *args, **kwargs) -> Chunk:
        pass

    def step(self, *args, **kwags):
        pass


class FIFOScheduler(ChunkScheduler):

    def __init__(self, rcache: MemoryPool, group: ChunkGroup) -> None:
        super().__init__(rcache, group)
        self.accessed_queue: Optional[deque] = None

    def reset(self) -> None:
        self.accessed_queue = deque()

    def clear(self) -> None:
        self.accessed_queue = None

    def upload_chunk(self, chunk: Chunk, *args, **kwargs) -> bool:
        self.accessed_queue.append(chunk)
        return True

    def offload_chunk(self, *args, **kwargs) -> Chunk:
        append_list = list()
        c = None
        while True:
            c = self.accessed_queue.popleft()
            if self.group.is_accessed(c):
                break
            else:
                append_list.append(c)
        for x in append_list:
            self.accessed_queue.append(x)
        return c
