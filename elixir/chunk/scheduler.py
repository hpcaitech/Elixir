from abc import ABC, abstractmethod
from typing import Optional

from .chunk import Chunk
from .group import ChunkGroup
from .memory_pool import MemoryPool, TensorBlock


class ChunkScheduler(ABC):

    def __init__(self, group: ChunkGroup) -> None:
        super().__init__()
        self.group = group
        self.releasable_set: Optional[set] = None
        self.current_step = 0

    @abstractmethod
    def reset(self) -> None:
        self.releasable_set = set()
        self.current_step = 0

    @abstractmethod
    def clear(self) -> None:
        # asure the set is empty now
        assert not bool(self.releasable_set)

    @abstractmethod
    def top(self, *args, **kwargs) -> Optional[Chunk]:
        # return None if the releasable set is empty
        if not self.releasable_set:
            return None

    @abstractmethod
    def add(self, chunk: Chunk, *args, **kwargs) -> None:
        if chunk in self.releasable_set:
            return
        self.releasable_set.add(chunk)

    @abstractmethod
    def remove(self, chunk: Chunk, *args, **kwargs) -> None:
        if chunk not in self.releasable_set:
            return
        self.releasable_set.remove(chunk)

    def step(self, *args, **kwags):
        self.current_step += 1


class FIFOScheduler(ChunkScheduler):

    def __init__(self, rcache: MemoryPool, group: ChunkGroup) -> None:
        super().__init__(rcache, group)
        self.fifo_dict: Optional[dict] = None

    def reset(self) -> None:
        super().reset()
        self.fifo_dict = dict()

    def clear(self) -> None:
        super().clear()
        self.fifo_dict = None

    def top(self, *args, **kwargs) -> Optional[Chunk]:
        super().top(*args, **kwargs)
        dict_iter = iter(self.fifo_dict)
        ret = next(dict_iter)
        return ret

    def add(self, chunk: Chunk, *args, **kwargs) -> None:
        super().add(chunk, *args, **kwargs)
        self.fifo_dict[chunk] = True

    def remove(self, chunk: Chunk, *args, **kwargs) -> None:
        super().remove(chunk, *args, **kwargs)
        self.fifo_dict.pop(chunk)
