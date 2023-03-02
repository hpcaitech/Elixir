import contextlib
import os
import random
from functools import cache

import numpy as np
import torch
import torch.distributed as dist

from . import meta_registrations


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


@cache
def gpu_dev():
    return torch.device(torch.cuda.current_device())


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:    # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def init_distributed():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])

    init_method = f'tcp://[{host}]:{port}'
    dist.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=world_size)

    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        torch.cuda.set_device(local_rank)

    seed_all(1024)
