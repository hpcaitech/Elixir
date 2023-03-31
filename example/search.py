from test.utils.gpt import GPTLMModel, small_data_fn

import torch
from transformers import AutoConfig, OPTConfig, OPTForCausalLM

import elixir
from elixir.ctx import MetaContext
from elixir.search import optimal_search
from elixir.search.utils import find_search_range
from elixir.tracer.memory_tracer import cuda_memory_profiling
from elixir.utils import get_model_size, model_size_formatter
from example.common.models import get_model
from example.common.utils import fake_gpt_data


def profile_optimal_search():
    elixir.cuda.set_memory_fraction(0.2)

    with MetaContext():
        model = get_model('opt-1b')
    model_size = get_model_size(model)
    print(f'model size: {model_size_formatter(model_size)}')

    ids, mask = fake_gpt_data(16, 1024, 50257)
    data = dict(input_ids=ids, attention_mask=mask)

    def train_step(model_in, inp_in):
        loss = model_in(**inp_in)
        loss.backward()

    model.gradient_checkpointing_enable()
    sr = optimal_search(model, 4, unified_dtype=torch.float16, overlap=True, verbose=True, inp=data, step_fn=train_step)


if __name__ == '__main__':
    profile_optimal_search()
