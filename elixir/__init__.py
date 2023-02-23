from functools import cache

import torch

from . import meta_registrations


@cache
def gpu_dev():
    return torch.device(torch.cuda.current_device())
