import torch
import torch.nn as nn


def to_divide(a: int, b: int):
    return a + (-a % b)


def get_multi_used_params(m: nn.Module) -> set[torch.Tensor]:
    multi_used_set = set()
    visit = dict()
    for module in m.modules():
        for param in module.parameters(recurse=False):
            if param not in visit:
                visit[param] = True
            else:
                multi_used_set.add(param)
    return multi_used_set
