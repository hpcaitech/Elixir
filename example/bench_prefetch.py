from functools import partial
from test.utils.gpt import GPTLMModel
from time import time

import colossalai
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

from elixir.search import simple_search
from elixir.utils import get_model_size, gpu_device, model_size_formatter
from elixir.wrapper import ElixirModule, ElixirOptimizer
from example.common.utils import fake_gpt_data, get_mem_info, get_profile_context, get_tflops, get_time_stamp


def resnet_tflops(model_numel, batch_size, step_time):
    return model_numel * batch_size * 6 / 1e9 / (step_time + 1e-12)


def benchmark_gpt(enable_prefetch: bool = True):
    batch_size = 32
    sequence_length = 1024
    vocab_size = 1024

    warmup_steps = 1
    number_steps = 6

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()

    global_size = dist.get_world_size()
    global_group = dist.GroupMember.WORLD

    model = GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, vocab_size=vocab_size)
    model = model.cuda()
    model.gradient_checkpointing_enable()

    model_size = get_model_size(model)
    print(f'model size: {model_size_formatter(model_size)}')

    optimizer = HybridAdam(model.parameters(), lr=1e-3)

    input_ids, attn_mask = fake_gpt_data(batch_size, sequence_length, vocab_size)
    input_dict = dict(input_ids=input_ids.cuda(), attention_mask=attn_mask.cuda())

    def fwd_bwd_step(local_model, local_input):
        local_model(**local_input).sum().backward()

    sr = simple_search(model,
                       global_size,
                       split_number=20,
                       allocate_factor=0.2,
                       shard_device=gpu_device(),
                       prefetch=enable_prefetch,
                       verbose=True,
                       inp=input_dict,
                       step_fn=fwd_bwd_step)
    model = ElixirModule(model, sr, global_group, prefetch=enable_prefetch)
    optimizer = ElixirOptimizer(model, optimizer)

    calc_tflops_func = partial(get_tflops, model_size, batch_size, sequence_length)
    torch.cuda.synchronize()
    model.train()
    tflops_list = []

    def train_step():
        step_start = time()
        loss = model(**input_dict)

        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - step_start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{number_steps}] Forward '), ranks=[0])

        optimizer.backward(loss)

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{number_steps}] Backward '), ranks=[0])

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - step_start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{number_steps}] Optimizer step '), ranks=[0])

        step_tflops = calc_tflops_func(step_time)
        logger.info(
            f'[{n + 1}/{number_steps}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {step_tflops:.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s',
            ranks=[0],
        )
        if n >= warmup_steps:
            tflops_list.append(step_tflops)

    is_profile = False
    demo_profiler = get_profile_context(is_profile,
                                        warmup_steps,
                                        number_steps - warmup_steps,
                                        save_dir=f'profile/{get_time_stamp()}-demo')

    with demo_profiler as prof:
        for n in range(number_steps):
            train_step()
            prof.step()

    tflops_list.sort()
    median_index = ((number_steps - warmup_steps) >> 1) + warmup_steps
    logger.info(f'Median TFLOPS is {tflops_list[median_index]:.3f}')
    torch.cuda.synchronize()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    benchmark_gpt()
