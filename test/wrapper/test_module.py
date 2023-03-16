import copy
import os
from functools import partial
from test.utils import TEST_MODELS, assert_dict_values

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from elixir.search import simple_search
from elixir.utils import init_distributed, seed_all
from elixir.wrapper import ElixirModule


def check_gradient(ddp_model: nn.Module, test_model: ElixirModule):
    name_to_grad = {name: param.grad for name, param in ddp_model.named_parameters()}
    param_to_name = {test_model.grad_state_dict[name]: name for name in name_to_grad}

    def check_chunks(cg, chunks):
        for c in chunks:
            cg.access_chunk(c)
            for t in c.get_tensors():
                name = param_to_name[t]
                grad = name_to_grad[name]

                torch.cuda.synchronize()
                print(f'checking the gradient of parameter `{name}`')
                assert_close(t.data, grad.data)

            if not c.rcache_fused:
                cg.release_chunk(c)

    check_chunks(test_model.param_chunk_group, test_model.param_chunk_group.fused_chunks)
    check_chunks(test_model.param_chunk_group, test_model.param_chunk_group.float_chunks)


def exam_module_init(nproc, group, grad_flag):
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('resnet')()
    torch_model = builder().cuda()
    for param in torch_model.parameters():
        param.requires_grad = grad_flag
    test_model = copy.deepcopy(torch_model)

    sr = simple_search(test_model, nproc)
    model = ElixirModule(test_model, sr, group)
    # check function: ElixirModule.state_dict after ElixirModule.__init__
    torch_st = torch_model.state_dict()
    test_st = model.state_dict()
    assert_dict_values(torch_st, test_st, fn=torch.equal)


def exam_one_module_fwd_bwd(builder, train_iter, nproc, group, exam_seed=2261):
    ddp_model = builder().cuda()
    test_model = copy.deepcopy(ddp_model)
    sr = simple_search(test_model, nproc, allocate_factor=0.6)
    test_model = ElixirModule(test_model, sr, group)

    # get different data
    seed_all(exam_seed + dist.get_rank(group))
    data, label = next(train_iter)
    data = data.cuda()

    seed_all(exam_seed, cuda_deterministic=True)
    ddp_model = DDP(ddp_model)
    ddp_loss = ddp_model(data).sum()
    ddp_loss.backward()

    test_loss = test_model(data).sum()
    test_model.backward(test_loss)

    assert_close(ddp_loss, test_loss)
    check_gradient(ddp_model.module, test_model)


def exam_modules_fwd_bwd(nproc, group):
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('small')()
    exam_one_module_fwd_bwd(builder, train_iter, nproc, group)


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    # exam_module_init(nproc=world_size, group=dist.GroupMember.WORLD, grad_flag=False)
    # exam_module_init(nproc=world_size, group=dist.GroupMember.WORLD, grad_flag=True)
    exam_modules_fwd_bwd(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_elixir_module(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_elixir_module(world_size=1)
