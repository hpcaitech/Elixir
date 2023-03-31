from time import time

import colossalai
import torch
import torch.distributed as dist

from elixir.utils import print_rank_0


def profile_function(n_times=20):
    l_gb = 10
    length = l_gb * 10**9
    cpu_x = torch.empty(length, dtype=torch.int8, pin_memory=True)
    cuda_x = torch.empty(length, dtype=torch.int8, device='cuda')

    torch.cuda.synchronize()
    cpu_to_cuda_start = time()
    for _ in range(n_times):
        cuda_x.copy_(cpu_x)
        torch.cuda.synchronize()
    cpu_to_cuda_span = time() - cpu_to_cuda_start

    cuda_to_cpu_start = time()
    for _ in range(n_times):
        cpu_x.copy_(cuda_x)
        torch.cuda.synchronize()
    cuda_to_cpu_span = time() - cuda_to_cpu_start

    n_proc = dist.get_world_size()
    sum_time = torch.tensor(cpu_to_cuda_span, dtype=torch.double, device='cuda')
    dist.all_reduce(sum_time, op=dist.ReduceOp.MAX)
    cpu_to_cuda_bandwidth = n_times * n_proc * l_gb / sum_time.item()

    sum_time = torch.tensor(cuda_to_cpu_span, dtype=torch.double, device='cuda')
    dist.all_reduce(sum_time, op=dist.ReduceOp.MAX)
    cuda_to_cpu_bandwidth = n_times * n_proc * l_gb / sum_time.item()

    print_rank_0(
        f'Bandwidth profiling result: cpu -> cuda: {cpu_to_cuda_bandwidth: .3f}, cuda -> cpu: {cuda_to_cpu_bandwidth: .3f}'
    )


if __name__ == '__main__':
    colossalai.launch_from_torch(config={})
    profile_function()
