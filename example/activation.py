import argparse
from test.utils.gpt import GPTLMModel, small_data_fn
from time import time

import torch
from torch.autograd.profiler_util import _format_memory
from transformers import AutoConfig, OPTConfig, OPTForCausalLM

from elixir.ctx import MetaContext
from elixir.kernels.attn_wrapper import wrap_attention
from elixir.tracer.memory_tracer import cuda_memory_profiling
from elixir.utils import get_model_size, model_size_formatter
from example.common.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='test activation settings')
    parser.add_argument('--model_name', type=str, default='opt-1b', help='test model name')
    args = parser.parse_args()
    return args


def profile_max_activation():
    args = parse_args()
    with MetaContext():
        model = get_model(args.model_name)
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
    print('activation', _format_memory(profiling_dict['activation_occ']))


if __name__ == '__main__':
    profile_max_activation()
