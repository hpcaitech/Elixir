def main():
    model = resnet18()

    max_numel = 0
    for name, param in model.named_parameters():
        max_numel = max(max_numel, param.numel())

    print(max_numel)

    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.cuda()

    pool = torch.empty((max_numel,), device='cuda', dtype=torch.float)
    for name, param in model.named_parameters():
        fake_data = pool[:param.numel()].view(param.shape)
        param.data = fake_data
        # print(name, param.shape, param.device)

    pre_max_cuda_memory = torch.cuda.memory_allocated()
    print(_format_memory(pre_max_cuda_memory))
    inp = torch.randn(4, 3, 32, 32, device='cuda')
    model(inp).sum().backward()

    aft_max_cuda_memory = torch.cuda.max_memory_allocated()
    print(_format_memory(aft_max_cuda_memory))

    print('activation space', aft_max_cuda_memory - pre_max_cuda_memory)
