from test.utils import TEST_MODELS

import torch
import torch.nn as nn

from elixir.tracer.param_tracer import generate_fx_order, generate_td_order


def test_td_forward():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('small')()
    model = builder()
    data, label = next(train_iter)

    # fx_forward_order = generate_fx_order(model)
    # for x in fx_forward_order:
    #     print(x)

    def foward_fn(model, inp):
        model(inp).sum().backward()

    td_forward_order = generate_td_order(model, data, foward_fn)
    for x in td_forward_order:
        print(x)


if __name__ == '__main__':
    test_td_forward()
