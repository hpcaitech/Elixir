from test.utils.iterator import TestIterator
from test.utils.mlp import MlpModule
from test.utils.registry import TEST_MODELS

import torch
import torch.nn as nn


class SmallIterator(TestIterator):

    def generate(self):
        data = torch.randint(low=0, high=20, size=(4, 8))
        label = torch.randint(low=0, high=2, size=(4,))
        return data, label


class SmallModule(nn.Module):

    def __init__(self, num_embeddings: int = 20, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpModule(hidden_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_embeddings, bias=False)
        self.proj.weight = self.embed.weight

    def forward(self, x):
        x = self.embed(x)
        x = x + self.norm1(self.mlp(x))
        x = self.proj(self.norm2(x))
        x = x.mean(dim=-2)
        return x


@TEST_MODELS.register('small')
def small_funcs():

    def model_builder():
        return SmallModule()

    train_iter = SmallIterator()
    valid_iter = SmallIterator()

    criterion = nn.CrossEntropyLoss()

    return model_builder, train_iter, valid_iter, criterion
