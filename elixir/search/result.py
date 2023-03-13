from typing import Dict, List

from elixir.chunk import ChunkGroup


class SearchResult(object):

    def __init__(self,
                 chunk_group: ChunkGroup,
                 chunk_config: List[Dict],
                 param_called_per_step: List[List[str]] = None) -> None:
        super().__init__()
        self.cg = chunk_group
        self.chunk_config_list = chunk_config
        self.param_called_per_step = param_called_per_step
