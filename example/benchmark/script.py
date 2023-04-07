import argparse
from functools import partial
from time import time

import torch
import torch.distributed as dist
from torch.autograd.profiler_util import _format_memory

import elixir
from elixir.utils import model_size_formatter, print_rank_0
from example.common.utils import fake_gpt_data, get_mem_info, get_profile_context, get_tflops, get_time_stamp


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark settings')
    parser.add_argument('--dp_type', type=str, default='fsdp', help='used ddp type in the benchmark')
    parser.add_argument('--memory_ratio',
                        type=float,
                        default=1.0,
                        help='the ratio of memory used for training, default is 1.0, which means using all the memory')
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2-400m',
        help='the name of the model to be benchmarked',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size per DP group of training.',
    )
    parser.add_argument(
        '--train_step',
        type=int,
        default=10,
        help='training iterations for test',
    )
    args = parser.parse_args()
    return args


def main():
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

    # distributed init
    elixir.utils.init_distributed()
    world_size = dist.get_world_size()
    print_rank_0(
        f'Benchmark Infomation: m_name={args.model_name}, n_gpu={world_size}, bs={args.batch_size}, type_dp={args.dp_type}'
    )

    elixir.cuda.set_memory_fraction(args.memory_ratio)
    print_rank_0(f'Resitrict cuda memory to {_format_memory(elixir.cuda.get_allowed_memory())}')

    torch.manual_seed(123)
    # we just use randomly generated data here
    input_ids, attn_mask = fake_gpt_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    data = dict(input_ids=input_ids.cuda(), attention_mask=attn_mask.cuda())

    # build GPT model
    if args.dp_type == 'fsdp':
        from example.common.fsdp import train_init
        fwd, bwd, opt, model_size = train_init(model_name=args.model_name)
    elif args.dp_type == 'elixir':
        from example.common.elx import train_init
        fwd, bwd, opt, model_size = train_init(model_name=args.model_name, data=data)
    elif args.dp_type.startswith('zero'):
        from example.common.ds import train_init

        if args.dp_type.startswith('zero2'):
            zero_stage = 2
        elif args.dp_type.startswith('zero3'):
            zero_stage = 3
        else:
            raise NotImplementedError

        if args.dp_type.endswith('offload'):
            cpu_offload = True
        else:
            cpu_offload = False

        fwd, bwd, opt, model_size = train_init(batch_size=args.batch_size,
                                               model_name=args.model_name,
                                               zero_stage=zero_stage,
                                               cpu_offload=cpu_offload)

    else:
        raise NotImplementedError
    print_rank_0(get_mem_info(prefix=f'After {args.dp_type} initialization: '))
    # print model size
    print_rank_0(f'the size of testing model size is {model_size_formatter(model_size)}.')

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, model_size, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()
    tflops_list = []

    def train_step():
        start = time()

        loss = fwd(data)

        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        print_rank_0(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '))

        bwd(loss)

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        print_rank_0(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '))

        opt()

        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        print_rank_0(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '))

        step_tflops = get_tflops_func(step_time)
        print_rank_0(
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, '
            f'TFLOPS: {step_tflops:.3f}, FWD time: {fwd_time:.3f}s, '
            f'BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s',)
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
    tflops_tensor = torch.tensor(tflops_list[median_index] / world_size, dtype=torch.float, device='cuda')
    dist.all_reduce(tflops_tensor)
    print_rank_0(f'Median TFLOPS is {tflops_tensor.item():.3f}')
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
