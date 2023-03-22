import torch
import torch.nn as nn


def to_divide(a: int, b: int):
    return a + (-a % b)


def to_meta_tensor(t: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
    # only float tensors need dtype change
    if t.is_floating_point() and dtype is not None:
        meta_dtype = dtype
    else:
        meta_dtype = t.dtype
    # we shall not use t.data.to here, since t might be a fake tensor
    meta_t = torch.empty(t.size(), dtype=meta_dtype, device='meta')
    # pack it if t is a parameter
    # we should filter parameters with no grad
    if isinstance(t, nn.Parameter) and t.requires_grad:
        meta_t = nn.Parameter(meta_t)
    return meta_t


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


def find_minimum_waste_size(numel_group_list: list[list[int]], min_range: int, max_range: int, interval: int):

    max_per_group = list()
    for n_list in numel_group_list:
        max_per_group.append(max(n_list))
    max_numel = max(max_per_group)

    test_size = to_divide(max(max_numel, min_range), interval)
    best_size = test_size
    min_waste = float('+inf')

    def calc_waste(numel_list: list[int], block_size: int):
        acc = 0
        left = 0
        for s in numel_list:
            if s > left:
                acc += left
                left = block_size
            left -= s
        return left + acc

    assert test_size <= max_range, 'max_numel or min_range is larger than max_range'
    while test_size <= max_range:
        current_waste = 0
        for n_list in numel_group_list:
            current_waste += calc_waste(n_list, test_size)
        if current_waste < min_waste:
            best_size = test_size
            min_waste = current_waste
        test_size += interval

    return best_size, min_waste
