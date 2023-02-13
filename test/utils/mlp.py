from test.utils.iterator import TestIterator
from test.utils.registry import TEST_MODELS

import torch
import torch.nn as nn


class MlpIterator(TestIterator):

    def generate(self):
        return torch.randn(4, 16), torch.randint(low=0, high=10, size=(4,))


class MlpModule(nn.Module):

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        return x + (self.proj2(self.act(self.proj1(x))))


@TEST_MODELS.register("mlp")
def mlp_funcs():

    def model_builder():
        return MlpModule()

    train_iter = MlpIterator()
    valid_iter = MlpIterator()

    criterion = nn.CrossEntropyLoss()

    return model_builder, train_iter, valid_iter, criterion
