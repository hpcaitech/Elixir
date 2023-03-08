import torch
import torch.nn as nn

from elixir.hook import HookParam
from elixir.parameter import FakeTensor


def test_hook():
    x = nn.Parameter(torch.randn(4, 4))

    ori_numel = x.numel()
    ori_size = x.size()
    ori_stride = x.stride()
    ori_offset = x.storage_offset()

    fake_data = FakeTensor(x.data)
    x.data = fake_data
    x.__class__ = HookParam

    assert x.numel() == ori_numel
    assert x.size() == ori_size
    assert x.stride() == ori_stride
    assert x.storage_offset() == ori_offset


if __name__ == '__main__':
    test_hook()
