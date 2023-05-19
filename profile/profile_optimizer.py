import os
from time import time

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.nn.optimizer import HybridAdam

from elixir.utils import print_rank_0


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split('hardware_concurrency() : ')[1]
    max_concurrency = int(inter_str.split('\n')[0])
    concurrency_per_process = max_concurrency // dist.get_world_size()
    os.environ['OMP_NUM_THREADS'] = str(concurrency_per_process)
    print(f'environmental variable OMP_NUM_THREADS is set to {max_concurrency}.')


class FlattenModel(nn.Module):

    def __init__(self, length: int = 2 * 10**9, device_type: str = 'cuda') -> None:
        super().__init__()
        self.length = length
        self.device_type = device_type
        self.weight = nn.Parameter(torch.zeros(length, device=device_type))

    def set_grad(self):
        self.weight.grad = torch.ones(self.length, dtype=torch.float, device=self.device_type)


def test_optimizer_update(device_type: str = 'cuda', n_times: int = 50):
    l_gb = 1
    length = int(l_gb * 10**9)
    model = FlattenModel(length, device_type=device_type)
    optimizer = HybridAdam(model.parameters(), lr=1e-5)

    sum_time = 0
    for _ in range(n_times):
        optimizer.zero_grad()
        model.set_grad()

        dist.barrier()
        torch.cuda.synchronize()

        start = time()
        optimizer.step()

        dist.barrier()
        torch.cuda.synchronize()
        sum_time += time() - start

    n_proc = dist.get_world_size()
    sum_time = torch.tensor(sum_time, dtype=torch.double, device='cuda')
    dist.all_reduce(sum_time, op=dist.ReduceOp.MAX)
    velocity = n_times * n_proc * l_gb / sum_time.item()

    print_rank_0(f'GPU velocity result: {velocity: .2f}')


if __name__ == '__main__':
    colossalai.launch_from_torch(config={})
    # set_cpu_maximum_parallelism()
    test_optimizer_update('cuda')
