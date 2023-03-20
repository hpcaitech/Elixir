from test.utils import TEST_MODELS

import pytest
import torch

from elixir.tracer.memory_tracer import cuda_memory_profiling


@pytest.mark.skip
def test_resnet_cuda():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('resnet')()

    model = builder().cuda()
    data, label = next(train_iter)
    data, label = data.cuda(), label.cuda()
    inp = (data, label)

    def train_step(model_in, inp_in):
        d_in, l_in = inp_in
        out = model_in(d_in)
        loss = criterion(out, l_in)
        loss.backward()

    train_step(model, inp)

    pre_cuda_alc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    train_step(model, inp)
    aft_cuda_alc = torch.cuda.max_memory_allocated()
    torch_activation_occ = aft_cuda_alc - pre_cuda_alc
    print('normal', torch_activation_occ)

    before = torch.cuda.memory_allocated()
    profiling_dict = cuda_memory_profiling(model, inp, train_step)
    after = torch.cuda.memory_allocated()
    print('profiling', profiling_dict)
    assert before == after
    assert torch_activation_occ == profiling_dict['activation_occ']
    print('Check is ok.')


@pytest.mark.skip
def test_gpt_cuda():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('gpt_small')()

    model = builder().cuda()
    model.gradient_checkpointing_enable()
    input_ids, attn_mask = next(train_iter)
    input_ids, attn_mask = input_ids.cuda(), attn_mask.cuda()
    inp = (input_ids, attn_mask)

    def train_step(model_in, inp_in):
        loss = model_in(*inp_in)
        loss.backward()

    train_step(model, inp)

    pre_cuda_alc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    train_step(model, inp)
    aft_cuda_alc = torch.cuda.max_memory_allocated()
    torch_activation_occ = aft_cuda_alc - pre_cuda_alc
    print('normal', torch_activation_occ)

    before = torch.cuda.memory_allocated()
    profiling_dict = cuda_memory_profiling(model, inp, train_step)
    after = torch.cuda.memory_allocated()
    print('profiling', profiling_dict)
    assert before == after
    assert torch_activation_occ == profiling_dict['activation_occ']
    print('Check is ok.')


if __name__ == '__main__':
    test_gpt_cuda()
