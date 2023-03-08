import torch
import torch.nn as nn

from elixir.parameter import OutplaceTensor


class ElixirModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
