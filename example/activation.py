from test.utils.gpt import GPTLMModel, small_data_fn
from time import time

import torch
from transformers import AutoConfig, OPTConfig, OPTForCausalLM

from elixir.ctx import MetaContext
from elixir.kernels.attn_wrapper import wrap_attention
from elixir.tracer.memory_tracer import cuda_memory_profiling
from elixir.utils import get_model_size, model_size_formatter
from example.common.models import get_model


def profile_max_activation():
    with MetaContext():
        model = get_model('opt-175b')
    model_size = get_model_size(model)
    print(f'model size: {model_size_formatter(model_size)}')

    data = small_data_fn()

    def train_step(model_in, inp_in):
        loss = model_in(**inp_in)
        loss.backward()

    model = wrap_attention(model)
    model.gradient_checkpointing_enable()

    start = time()

    profiling_dict = cuda_memory_profiling(model, data, train_step, dtype=torch.float16)

    torch.cuda.synchronize()
    end = time()

    print(f'profile time: {end - start: .2f} sec')
    print('memory usage', profiling_dict)


if __name__ == '__main__':
    profile_max_activation()
