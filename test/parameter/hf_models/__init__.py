from copy import deepcopy

import torch


def test_hf_model(builder, kwargs, data_fn):
    torch_model = builder(**kwargs).cuda()
    test_model = deepcopy(torch_model)

    from elixir.parameter.temp import transform
    test_model = transform(test_model)

    torch_model.eval()
    test_model.eval()

    data = data_fn()
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()

    torch_out = torch_model(**data)
    test_out = test_model(**data)

    for k, u in torch_out.items():
        v = test_out[k]
        if isinstance(u, torch.Tensor):
            assert torch.equal(u, v), f'output {k} is wrong'

    torch.cuda.synchronize()
