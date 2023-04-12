from functools import partial
from test.utils import to_cuda

import diffusers
import torch
import torch.distributed as dist
import torch.nn as nn
from torchaudio.models import hubert_base, wav2vec2_base

from elixir.search import minimum_waste_search
from elixir.utils import init_distributed
from elixir.wrapper import ElixirModule


def wav2vec_data_gen_fn():
    batch_size, num_frames = 4, 400
    waveforms = torch.randn(batch_size, num_frames)
    lengths = torch.randint(0, num_frames, (batch_size,))
    return dict(waveforms=waveforms, lengths=lengths)


data_vae_fn = lambda: dict(sample=torch.randn(2, 3, 32, 32))


def run(model_fn, data_fn):
    world_size = dist.get_world_size()
    world_group = dist.GroupMember.WORLD

    model = model_fn().cuda()
    # sr = minimum_waste_search(model, world_size, verbose=True)
    # model = ElixirModule(model, sr, world_group)

    data = to_cuda(data_fn())
    output = model(**data)['sample']
    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name, type(param.grad))

    torch.cuda.synchronize()


def profile(model_fn, data_fn):
    from elixir.tracer.param_tracer import generate_td_order
    model = model_fn()
    data = data_fn()
    data = (data['x'],)

    def forward_backward_fn(model, inp):
        model(*inp).sum().backward()

    td_order = generate_td_order(model, data, forward_backward_fn)
    print(td_order)


if __name__ == '__main__':
    init_distributed()
    run(diffusers.VQModel, data_vae_fn)
