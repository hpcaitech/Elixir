from copy import deepcopy

import timm.models as tmm
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from elixir.parameter.temp import transform
from elixir.utils import seed_all


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
    test_model = transform(test_model)

    torch_model.eval()
    test_model.eval()

    torch_out = torch_model(data)
    test_out = test_model(data)

    assert_tuple_close(torch_out, test_out)
    torch.cuda.synchronize()


def test_timm_models():
    seed_all(1001, cuda_deterministic=True)
    model_list = [
        tmm.beit_base_patch16_224,
        tmm.beitv2_base_patch16_224,
        tmm.cait_s24_224,
        tmm.coat_lite_mini,
        tmm.convit_base,
        tmm.deit3_base_patch16_224,
        tmm.dm_nfnet_f0,
        tmm.eca_nfnet_l0,
        tmm.efficientformer_l1,
        tmm.ese_vovnet19b_dw,
        tmm.gmixer_12_224,
        tmm.gmlp_b16_224,
        tmm.hardcorenas_a,
        tmm.hrnet_w18_small,
        tmm.inception_v3,
        tmm.mixer_b16_224,
        tmm.nf_ecaresnet101,
        tmm.nf_regnet_b0,
    # tmm.pit_b_224,  # pretrained only
        tmm.regnetv_040,
        tmm.skresnet18,
        tmm.swin_base_patch4_window7_224,
        tmm.tnt_b_patch16_224,
        tmm.vgg11,
        tmm.vit_base_patch16_18x2_224,
        tmm.wide_resnet50_2,
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
    test_timm_models()
    # torch.Tensor.add_ = torch.Tensor.add
    # test_model(tm.resnest.resnest50d, {})
