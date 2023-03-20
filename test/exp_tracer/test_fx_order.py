from test.utils import TEST_MODELS, assert_dict_keys

import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_fx_order


def test_fx_forward():
    builder, *_ = TEST_MODELS.get_func('small')()
    model = builder()
    forward_order = generate_fx_order(model)

    # for step in forward_order:
    #     print(step)

    assert_dict_keys(forward_order[0], ['embed.weight'])
    assert_dict_keys(forward_order[1], ['mlp.proj1.weight', 'mlp.proj1.bias'])
    assert_dict_keys(forward_order[2], ['mlp.proj2.weight', 'mlp.proj2.bias'])
    assert_dict_keys(forward_order[3], ['norm1.weight', 'norm1.bias'])
    assert_dict_keys(forward_order[4], ['norm2.weight', 'norm2.bias'])
    assert_dict_keys(forward_order[5], ['embed.weight'])


if __name__ == '__main__':
    test_fx_forward()
