from test.utils import TEST_MODELS, assert_dict_keys

import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_tf_order


class M(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(16)

    def forward(self, x):
        return x + self.bn(x)


def test_tf_forward_backward():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('resnet')()
    model = builder()
    data, label = next(train_iter)

    # model = M()
    # data = torch.randn(4, 16)

    def forward_backward_fn(model, data):
        model(data).sum().backward()

    tf_order = generate_tf_order(model, data, forward_backward_fn)
    for step_dict in tf_order:
        print(step_dict)

    exit(0)

    assert_dict_keys(td_order[0], ['proj1.weight'])
    assert_dict_keys(td_order[1], ['proj1.weight', 'proj1.bias'])
    assert_dict_keys(td_order[2], ['proj2.weight'])
    assert_dict_keys(td_order[3], ['proj2.weight', 'proj2.bias'])
    assert_dict_keys(td_order[4], ['proj2.weight'])
    assert_dict_keys(td_order[5], ['proj2.weight'])
    assert_dict_keys(td_order[6], ['proj1.weight'])
    assert_dict_keys(td_order[7], ['proj1.weight'])


if __name__ == '__main__':
    test_tf_forward_backward()
