import os
from functools import partial
from time import time

import colossalai
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

from elixir.search import minimum_waste_search
from elixir.utils import get_model_size, gpu_device, model_size_formatter
from elixir.wrapper import ElixirModule, ElixirOptimizer
from example.common.models import get_model
from example.common.utils import fake_gpt_data, get_mem_info, get_profile_context, get_tflops, get_time_stamp


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size per DP group of training.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2-400m',
        help='model model scale',
    )
    parser.add_argument(
        '--train_step',
        type=int,
        default=10,
        help='training iterations for test',
    )

    args = parser.parse_args()
    return args


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split('hardware_concurrency() : ')[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ['OMP_NUM_THREADS'] = max_concurrency
    print(f'environmental variable OMP_NUM_THREADS is set to {max_concurrency}.')


def main():
    set_cpu_maximum_parallelism()
    args = parse_args()

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, 'warmup steps should smaller than the total steps'
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, 'the number of valid steps should be odd to take the median'
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f'{args.model_name}, batch size {BATCH_SIZE}', ranks=[0])

    torch.manual_seed(123)

    # build GPT model
    model = get_model(args.model_name)
    model.gradient_checkpointing_enable()

    # build a highly optimized gpu/cpu optimizer
    optimizer = HybridAdam(model.parameters(), lr=1e-3)

    input_ids, attn_mask = fake_gpt_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    input_dict = dict(input_ids=input_ids.cuda(), attention_mask=attn_mask.cuda())

    def fwd_bwd_step(local_model, local_input):
        local_model(**local_input).sum().backward()

    global_size = dist.get_world_size()
    global_group = dist.GroupMember.WORLD

    sr = minimum_waste_search(
    # pre-commit: do not rearrange
        m=model,
        group_size=global_size,
        unified_dtype=torch.float16,
        prefetch=True,
        verbose=True,
        inp=input_dict,
        step_fn=fwd_bwd_step)

    # wrap your model and optimizer
    model = ElixirModule(model, sr, global_group, prefetch=True, dtype=torch.float16)
    optimizer = ElixirOptimizer(model, optimizer, initial_scale=32)

    logger.info(get_mem_info(prefix='After Elixir initialization: '), ranks=[0])

    # print model size
    numel = get_model_size(model)
    logger.info(f'the size of testing model size is {model_size_formatter(numel)}.')

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    # we just use randomly generated data here
    input_ids, attn_mask = fake_gpt_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    input_ids, attn_mask = input_ids.cuda(), attn_mask.cuda()

    torch.cuda.synchronize()
    model.train()
    tflops_list = []

    def train_step():
        start = time()
        loss = model(input_ids, attn_mask)

        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        optimizer.backward(loss)

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {step_tflops:.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s',
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    demo_profiler = get_profile_context(PROF_FLAG,
                                        WARMUP_STEPS,
                                        NUM_STEPS - WARMUP_STEPS,
                                        save_dir=f'profile/{get_time_stamp()}-demo')

    with demo_profiler as prof:
        for n in range(NUM_STEPS):
            train_step()
            prof.step()

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f'Median TFLOPS is {tflops_list[median_index]:.3f}')
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
