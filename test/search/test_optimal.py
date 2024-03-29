from copy import deepcopy
from test.utils import TEST_MODELS

import torch

from elixir.cuda import gpu_device
from elixir.search import optimal_search


def step_fn(model, inp):
    model(**inp).backward()


def test_optimal_search():
    model_fn, data_fn = TEST_MODELS.get('gpt2_small')
    model = model_fn()
    data = data_fn()

    sr = optimal_search(model, 1, unified_dtype=torch.float16, overlap=True, verbose=True, inp=data, step_fn=step_fn)

    chunk_plans = deepcopy(sr.param_chunk_plans)
    for plan in chunk_plans:
        assert plan.chunk_dtype == torch.float16
        assert plan.kwargs.get('shard_device') == gpu_device()


if __name__ == '__main__':
    test_optimal_search()
