import copy
import os
from functools import partial
from test.utils import TEST_MODELS, assert_dict_values

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from elixir import init_distributed, seed_all
from elixir.search import simple_search
from elixir.wrapper import ElixirModule


def exam_module_init(nproc, group):
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('resnet')()
    torch_model = builder()
    test_model = copy.deepcopy(torch_model)
    test_model = test_model.cuda()

    sr = simple_search(test_model, nproc, 10)
    model = ElixirModule(test_model, sr, group)

    torch_st = torch_model.state_dict()
    test_st = model.state_dict()
    assert_dict_values(torch_st, test_st, fn=torch.equal)


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_module_init(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
def test_elixir_module():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_elixir_module()
