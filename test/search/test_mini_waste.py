from copy import deepcopy
from test.utils import TEST_MODELS

import torch

from elixir.search import minimum_waste_search
from elixir.utils import gpu_device


def step_fn(model, inp):
    model(**inp).sum().backward()


def test_mini_waste_search():
    builder, data_iter, *_ = TEST_MODELS.get_func('gpt_small')()
    model = builder()

    input_ids, attn_mask = next(data_iter)
    example_input = dict(input_ids=input_ids, attention_mask=attn_mask)

    sr = minimum_waste_search(model,
                              1,
                              unified_dtype=torch.float16,
                              cpu_offload=True,
                              prefetch=True,
                              verbose=True,
                              inp=example_input,
                              step_fn=step_fn)

    chunk_plans = deepcopy(sr.param_chunk_plans)
    for plan in chunk_plans:
        assert plan.chunk_dtype == torch.float16
        assert plan.kwargs.get('shard_device') == torch.device('cpu')
        assert plan.kwargs.get('cpu_pin_memory') == True


if __name__ == '__main__':
    test_mini_waste_search()
