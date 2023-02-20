from test.utils import TEST_MODELS, assert_dict_keys

import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_fx_order, generate_td_order


def test_td_forward_backward():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('mlp')()
    model = builder()
    data, label = next(train_iter)
    data.requires_grad = True

    def forward_backward_fn(model, inp):
        model(*inp).sum().backward()

    td_order = generate_td_order(model, data, forward_backward_fn)

    assert_dict_keys(td_order[0], ['proj1.weight'])
    assert_dict_keys(td_order[1], ['proj1.weight', 'proj1.bias'])
    assert_dict_keys(td_order[2], ['proj2.weight'])
    assert_dict_keys(td_order[3], ['proj2.weight', 'proj2.bias'])
    assert_dict_keys(td_order[4], ['proj2.weight'])
    assert_dict_keys(td_order[5], ['proj2.weight'])
    assert_dict_keys(td_order[6], ['proj1.weight'])
    assert_dict_keys(td_order[7], ['proj1.weight'])


if __name__ == '__main__':
    test_td_forward_backward()
