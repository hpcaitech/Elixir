from abc import ABC, abstractmethod
from typing import Optional

from .chunk import Chunk
from .group import ChunkGroup
from .memory_pool import MemoryPool, TensorBlock


class ChunkScheduler(ABC):

    def __init__(self) -> None:
        super().__init__()
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
    def top(self) -> Optional[Chunk]:
        # return None if the releasable set is empty
        if not self.releasable_set:
            return False
        return True

    @abstractmethod
    def add(self, chunk: Chunk) -> bool:
        if chunk in self.releasable_set:
            return False
        self.releasable_set.add(chunk)
        return True

    @abstractmethod
    def remove(self, chunk: Chunk) -> bool:
        if chunk not in self.releasable_set:
            return False
        self.releasable_set.remove(chunk)
        return True

    def step(self, *args, **kwags):
        self.current_step += 1


class FIFOScheduler(ChunkScheduler):

    def __init__(self) -> None:
        super().__init__()
        self.fifo_dict: Optional[dict] = None

    def reset(self) -> None:
        super().reset()
        self.fifo_dict = dict()

    def clear(self) -> None:
        super().clear()
        self.fifo_dict = None

    def top(self) -> Optional[Chunk]:
        if not super().top():
            return None
        dict_iter = iter(self.fifo_dict)
        ret = next(dict_iter)
        return ret

    def add(self, chunk: Chunk) -> bool:
        if not super().add(chunk):
            return False
        self.fifo_dict[chunk] = True
        return True

    def remove(self, chunk: Chunk) -> bool:
        if not super().remove(chunk):
            return False
        self.fifo_dict.pop(chunk)
        return True
