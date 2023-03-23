from test.utils import to_cuda

import pytest
import torch


def test_registry():
    from test.utils.registry import TEST_MODELS
    for name, model_tuple in TEST_MODELS:
        torch.cuda.synchronize()
        print(f'model `{name}` is in testing')

        model_fn, data_fn = model_tuple
        model = model_fn().cuda()
        data = to_cuda(data_fn())
        loss = model(**data)
        loss.backward()


if __name__ == '__main__':
    test_registry()
