import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_fx_order


class EModule(nn.Module):

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.empty(4, 4))
        self.proj1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        a = self.u
        b = a
        x = x + b
        c = self.u
        y = x + (self.proj2(self.act(self.proj1(x))))
        return c + y


def test_fx_forward():
    model = EModule()
    forward_order = generate_fx_order(model)

    key0 = forward_order[0].keys()
    assert len(key0) == 1
    assert 'u' in key0

    key1 = forward_order[1].keys()
    assert len(key1) == 2
    assert 'proj1.weight' in key1
    assert 'proj1.bias' in key1

    key2 = forward_order[2].keys()
    assert len(key2) == 2
    assert 'proj2.weight' in key2
    assert 'proj2.bias' in key2

    key3 = forward_order[3].keys()
    assert len(key3) == 1
    assert 'u' in key3


if __name__ == '__main__':
    test_fx_forward()
