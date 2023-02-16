from test.utils import TEST_MODELS

import torch

from elixir.tracer.memory_tracer import cuda_memory_profiling


def test_cuda():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('mlp')()

    model = builder().cuda()
    data, label = next(train_iter)
    data, label = data.cuda(), label.cuda()
    inp = (data, label)

    def train_step(model_in, inp_in):
        d_in, l_in = inp_in
        out = model_in(d_in)
        loss = criterion(out, l_in)
        loss.backward()

    torch_param_occ = 0
    for name, param in model.named_parameters():
        torch_param_occ += param.numel() * param.element_size()

    torch_buffer_occ = 0
    for name, buffer in model.named_buffers():
        torch_buffer_occ += buffer.numel() * buffer.element_size()

    pre_cuda_alc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    train_step(model, inp)
    aft_cuda_alc = torch.cuda.max_memory_allocated()
    torch_activation_occ = aft_cuda_alc - pre_cuda_alc
    print(torch_param_occ, torch_buffer_occ, torch_activation_occ)
    profiling_dict = cuda_memory_profiling(model, inp, train_step)
    print(profiling_dict)


if __name__ == '__main__':
    test_cuda()
