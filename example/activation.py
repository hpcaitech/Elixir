from test.utils.gpt import GPTLMModel, MacroIterator

import torch
from transformers import AutoConfig, OPTConfig, OPTForCausalLM

from elixir import get_model_size, model_size_formatter
from elixir.ctx import MetaContext
from elixir.tracer.memory_tracer import cuda_memory_profiling


def gpt2_10b():
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16)


def gpt2_20b():
    return GPTLMModel(hidden_size=8192, num_layers=25, num_attention_heads=16)


def gpt2_24b():
    return GPTLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16)


def gpt2_30b():
    return GPTLMModel(hidden_size=8192, num_layers=37, num_attention_heads=16)


def gpt2_40b():
    return GPTLMModel(hidden_size=8192, num_layers=50, num_attention_heads=16)


def profile_gpt():
    with MetaContext():
        model = gpt2_40b()
    model_size = get_model_size(model)
    print(f'model size: {model_size_formatter(model_size)}')

    train_iter = MacroIterator()
    input_ids, attn_mask = next(train_iter)
    input_ids, attn_mask = input_ids.cuda(), attn_mask.cuda()
    inp = (input_ids, attn_mask)

    def train_step(model_in, inp_in):
        loss = model_in(*inp_in)
        loss.backward()

    model.gradient_checkpointing_enable()
    profiling_dict = cuda_memory_profiling(model, inp, train_step, dtype=torch.float16)
    print('memory usage', profiling_dict)


def profile_opt():
    # config = AutoConfig.from_pretrained('facebook/opt-66b')
    # OPT-175B configuration
    config = OPTConfig(activation_dropout=0.0,
                       hidden_size=12288,
                       num_hidden_layers=96,
                       ffn_dim=49152,
                       num_attention_heads=96,
                       word_embed_proj_dim=12288,
                       output_projection=True)

    with MetaContext():
        model = OPTForCausalLM(config)
    model_size = get_model_size(model)
    print(f'model size: {model_size_formatter(model_size)}')

    train_iter = MacroIterator()
    input_ids, attn_mask = next(train_iter)
    input_ids, attn_mask = input_ids.cuda(), attn_mask.cuda()
    inp = (input_ids, attn_mask)

    def train_step(model_in, inp_in):
        input_ids, attn_mask = inp_in
        outputs = model_in(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        loss = outputs['loss']
        loss.backward()

    model.gradient_checkpointing_enable()
    profiling_dict = cuda_memory_profiling(model, inp, train_step, dtype=torch.float16)
    print('memory usage', profiling_dict)


if __name__ == '__main__':
    profile_gpt()
