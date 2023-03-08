import copy
import os
from functools import partial
from test.utils import TEST_MODELS

import pytest
import torch
import torch.distributed as dist

from elixir import init_distributed, seed_all
from elixir.hook import hook_transform


def exam_chunk_fetcher(nproc, group):
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('mlp')()
    torch_model = builder().cuda()
    test_model = copy.deepcopy(torch_model)

    rank = dist.get_rank(group)
    # get different data
    seed_all(1001 + rank)
    data, label = next(train_iter)
    data = data.cuda()

    hook_model = hook_transform(test_model, group)
    hook_model(data).sum().backward()


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_chunk_fetcher(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
def test_chunk_fetcher():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_fetcher()
