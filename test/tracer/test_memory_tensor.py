import pytest
import torch

from elixir.tracer.memory_tracer import MTensor


@pytest.mark.parametrize('dtype', [torch.float, torch.float16, torch.bfloat16])
def test_mm(dtype, size0=(4, 256), size1=(256, 1024)):
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.memory_allocated() == 0

    x = torch.randn(size0, dtype=dtype, device='cuda')
    y = torch.randn(size1, dtype=dtype, device='cuda')
    torch_pre_alc = torch.cuda.memory_allocated()

    z = torch.matmul(x, y)
    torch_z_size = z.shape
    torch_act_alc = torch.cuda.max_memory_allocated() - torch_pre_alc

    del x
    del y
    del z

    assert torch.cuda.memory_allocated() == 0
    x = MTensor(torch.randn(size0, dtype=dtype, device='cuda'))
    y = MTensor(torch.randn(size1, dtype=dtype, device='cuda'))
    mtensor_pre_alc = torch.cuda.memory_allocated()

    z = torch.matmul(x, y)
    mtensor_z_size = z.shape
    mtensor_act_alc = torch.cuda.max_memory_allocated() - mtensor_pre_alc

    assert torch_z_size == mtensor_z_size
    assert torch_pre_alc == mtensor_pre_alc
    assert torch_act_alc == mtensor_act_alc


@pytest.mark.parametrize('dtype', [torch.float, torch.float16, torch.bfloat16])
def test_addmm(dtype, size0=(4, 16), size1=(16, 64)):
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.memory_allocated() == 0

    x = torch.randn(size0, dtype=dtype, device='cuda')
    y = torch.randn(size1, dtype=dtype, device='cuda')
    u = torch.randn(size1[-1], dtype=dtype, device='cuda')
    torch_pre_alc = torch.cuda.memory_allocated()

    z = torch.addmm(u, x, y)
    torch_z_size = z.shape
    torch_act_alc = torch.cuda.max_memory_allocated() - torch_pre_alc

    del x
    del y
    del u
    del z

    assert torch.cuda.memory_allocated() == 0
    x = MTensor(torch.randn(size0, dtype=dtype, device='cuda'))
    y = MTensor(torch.randn(size1, dtype=dtype, device='cuda'))
    u = MTensor(torch.randn(size1[-1], dtype=dtype, device='cuda'))
    mtensor_pre_alc = torch.cuda.memory_allocated()

    z = torch.addmm(u, x, y)
    mtensor_z_size = z.shape
    mtensor_act_alc = torch.cuda.max_memory_allocated() - mtensor_pre_alc

    assert torch_z_size == mtensor_z_size
    assert torch_pre_alc == mtensor_pre_alc
    assert torch_act_alc == mtensor_act_alc


if __name__ == '__main__':
    test_addmm(dtype=torch.float)
