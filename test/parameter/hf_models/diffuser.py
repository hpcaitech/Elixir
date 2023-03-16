from copy import deepcopy
from test.parameter.hf_models import test_hf_model

import diffusers
import torch
from torch.testing import assert_close

from elixir.parameter.temp import transform
from elixir.utils import seed_all

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)
TIME_STEP = 3


def test_model(builder, kwargs):
    data = torch.randn(LATENTS_SHAPE, device='cuda')
    torch_model = builder().cuda()
    test_model = deepcopy(torch_model)
    test_model = transform(test_model)

    torch_model.eval()
    test_model.eval()

    torch_out = torch_model(data, **kwargs)
    test_out = test_model(data, **kwargs)

    assert_close(torch_out['sample'], test_out['sample'])
    torch.cuda.synchronize()


def test_vae():
    seed_all(319)
    model_list = [
        diffusers.AutoencoderKL,
        diffusers.VQModel,
    ]
    kwargs = {}
    for builder in model_list:
        flag = '√'
        try:
            test_model(builder, kwargs)
        except:
            flag = 'x'
        print(f'{builder.__name__:40s} {flag}')


def test_unet():
    seed_all(221)
    model_list = [
        diffusers.UNet2DModel,
    ]
    kwargs = {'timestep': TIME_STEP}
    for builder in model_list:
        flag = '√'
        try:
            test_model(builder, kwargs)
        except:
            flag = 'x'
        print(f'{builder.__name__:40s} {flag}')


if __name__ == '__main__':
    test_vae()
    test_unet()
