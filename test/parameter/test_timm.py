from copy import deepcopy

import timm.models as tm
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from elixir import seed_all
from elixir.parameter.temp import transform


def assert_tuple_close(ta, tb):
    if not isinstance(ta, tuple):
        ta = (ta,)
    if not isinstance(tb, tuple):
        tb = (tb,)

    for (a, b) in zip(ta, tb):
        assert_close(a, b)


def test_model(builder, kwargs):
    data = torch.randn(2, 3, 224, 224, device='cuda')
    torch_model = builder(**kwargs).cuda()
    test_model = deepcopy(torch_model)

    torch_model.eval()
    test_model.eval()
    test_model = transform(test_model, concrete_args={'x': data})

    torch_out = torch_model(data)
    test_out = test_model(data)

    assert_tuple_close(torch_out, test_out)
    torch.cuda.synchronize()


def test_timm_models():
    seed_all(1001, cuda_deterministic=True)
    model_list = [
        tm.resnest.resnest50d, tm.beit.beit_base_patch16_224, tm.cait.cait_s24_224, tm.convmixer.convmixer_768_32,
        tm.efficientnet.efficientnetv2_m, tm.resmlp_12_224, tm.vision_transformer.vit_base_patch16_224,
        tm.deit_base_distilled_patch16_224, tm.convnext.convnext_base, tm.vgg.vgg11, tm.dpn.dpn68,
        tm.densenet.densenet121, tm.rexnet.rexnet_100, tm.swin_transformer.swin_base_patch4_window7_224
    ]

    for builder in model_list:
        kwargs = {}

        flag = '√'
        try:
            test_model(builder, kwargs)
        except:
            flag = 'x'
        print(f'{builder.__name__:40s} {flag}')


if __name__ == '__main__':
    # test_timm_models()
    torch.Tensor.add_ = torch.Tensor.add
    test_model(tm.resnest.resnest50d, {})
