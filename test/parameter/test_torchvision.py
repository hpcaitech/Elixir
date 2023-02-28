from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.models as tm
from torch.testing import assert_close

from elixir import seed_all
from elixir.parameter.temp import transform


def test_model(builder, kwargs):
    torch_model = builder(**kwargs).cuda()
    test_model = deepcopy(torch_model)
    test_model = transform(test_model)

    torch_model.eval()
    test_model.eval()

    data = torch.randn(2, 3, 224, 224, device='cuda')
    torch_out = torch_model(data)
    test_out = test_model(data)
    assert_close(torch_out, test_out)


def test_torchvision_models():
    seed_all(1001, cuda_deterministic=True)
    model_list = [
        tm.vgg11, tm.resnet18, tm.densenet121, tm.mobilenet_v3_small, tm.resnext50_32x4d, tm.wide_resnet50_2,
        tm.regnet_x_16gf, tm.mnasnet0_5, tm.efficientnet_b0, tm.vit_b_16, tm.convnext_small
    ]
    rand_list = [tm.efficientnet_b0, tm.convnext_small]
    for builder in model_list:
        kwargs = {}
        if builder in rand_list:
            kwargs['stochastic_depth_prob'] = 0

        flag = '√'
        try:
            test_model(builder, kwargs)
        except:
            flag = 'x'

        print(f'{builder.__name__:20s} {flag}')


if __name__ == '__main__':
    test_torchvision_models()
