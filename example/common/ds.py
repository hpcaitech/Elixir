import json
import os

import deepspeed
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam

from example.common.models import get_model


def train_init(batch_size: int, model_name: str, zero_stage: int, cpu_offload: bool):
    if zero_stage == 2:
        ds_path = './zero2_config.json'
    else:
        ds_path = './zero3_config.json'
    ds_config = json.load(open(ds_path))

    if not cpu_offload:
        zero_optim = ds_config.get('zero_optimization')
        zero_optim.pop('offload_optimizer')
        if zero_stage == 3:
            zero_optim.pop('offload_param')

    total_bs = batch_size * int(os.environ['WORLD_SIZE'])
    ds_config['train_batch_size'] = total_bs
    ds_config['train_micro_batch_size_per_gpu'] = batch_size

    deepspeed.init_distributed()
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = get_model(model_name)
    numel = deepspeed.runtime.zero.partition_parameters.param_count

    if cpu_offload:
        optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

    def forward(data):
        return model(**data)

    def backward(loss):
        model.backward(loss)

    def optim():
        model.step()

    return forward, backward, optim, numel


if __name__ == '__main__':
    train_init(1, 'opt-1b', 3, True)
    exit(0)
