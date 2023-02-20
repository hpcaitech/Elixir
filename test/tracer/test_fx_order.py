from test.utils import TEST_MODELS

import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_fx_order


def test_fx_forward():
    builder, *_ = TEST_MODELS.get_func('small')()
    model = builder()
    forward_order = generate_fx_order(model)

    # for step in forward_order:
    #     print(step)

    key0 = forward_order[0].keys()
    assert len(key0) == 1
    assert 'embed.weight' in key0

    key1 = forward_order[1].keys()
    assert len(key1) == 2
    assert 'mlp.proj1.weight' in key1
    assert 'mlp.proj1.bias' in key1

    key2 = forward_order[2].keys()
    assert len(key2) == 2
    assert 'mlp.proj2.weight' in key2
    assert 'mlp.proj2.bias' in key2

    key3 = forward_order[3].keys()
    assert len(key3) == 2
    assert 'norm1.weight' in key3
    assert 'norm1.bias' in key3

    key4 = forward_order[4].keys()
    assert len(key4) == 2
    assert 'norm2.weight' in key4
    assert 'norm2.bias' in key4

    key5 = forward_order[5].keys()
    assert len(key5) == 1
    assert 'embed.weight' in key5


if __name__ == '__main__':
    test_fx_forward()
