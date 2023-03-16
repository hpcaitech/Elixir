from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.models as tm
from torch.testing import assert_close

from elixir.parameter.temp import transform
from elixir.utils import seed_all


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
        tm.alexnet,
        tm.convnext_base,
        tm.densenet121,
        tm.efficientnet_v2_s,
        tm.googlenet,    # output bad case
        tm.inception_v3,    # bad case
        tm.mobilenet_v2,
        tm.mobilenet_v3_small,
        tm.mnasnet0_5,
        tm.resnet18,
        tm.regnet_x_16gf,
        tm.resnext50_32x4d,
        tm.shufflenet_v2_x0_5,
        tm.squeezenet1_0,
        tm.swin_s,    # fx bad case
        tm.vgg11,
        tm.vit_b_16,
        tm.wide_resnet50_2,
    ]
    for builder in model_list:
        kwargs = {}
        flag = '√'
        try:
            test_model(builder, kwargs)
        except:
            flag = 'x'

        print(f'{builder.__name__:20s} {flag}')


def test_fwd_bwd(builder, kwargs):
    torch_model = builder(**kwargs).cuda()
    test_model = deepcopy(torch_model)
    test_model = transform(test_model)

    torch_model.eval()
    test_model.eval()

    data = torch.randn(2, 3, 224, 224, device='cuda')
    torch_loss = torch_model(data).sum()
    torch_loss.backward()

    test_loss = test_model(data).sum()
    assert_close(torch_loss, test_loss)
    test_loss.backward()

    for (torch_p, test_p) in zip(torch_model.parameters(), test_model.parameters()):
        assert_close(torch_p.grad, test_p.grad)


def test_fwd_bwd_models():
    seed_all(1001, cuda_deterministic=True)

    model_list = [
        tm.alexnet,
        tm.convnext_base,
        tm.densenet121,
        tm.efficientnet_v2_s,
        tm.googlenet,    # output bad case
        tm.inception_v3,    # bad case
        tm.mobilenet_v2,
        tm.mobilenet_v3_small,
        tm.mnasnet0_5,
        tm.resnet18,
        tm.regnet_x_16gf,
        tm.resnext50_32x4d,
        tm.shufflenet_v2_x0_5,
        tm.squeezenet1_0,
        tm.swin_s,    # fx bad case
        tm.vgg11,
        tm.vit_b_16,
        tm.wide_resnet50_2,
    ]
    for builder in model_list:
        kwargs = {}
        flag = '√'
        try:
            test_fwd_bwd(builder, kwargs)
        except:
            flag = 'x'

        print(f'{builder.__name__:20s} {flag}')


if __name__ == '__main__':
    test_fwd_bwd_models()
