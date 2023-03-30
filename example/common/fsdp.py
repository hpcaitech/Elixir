import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, ShardingStrategy

from elixir.ctx import MetaContext
from elixir.utils import get_model_size
from example.common.models import get_model


def train_init(model_name: str):
    global_group = dist.GroupMember.WORLD
    global_rank = dist.get_rank()

    with MetaContext('cuda'):
        model = get_model(model_name)
    model_size = get_model_size(model)

    model = FSDP(module=model,
                 process_group=global_group,
                 device_id=global_rank,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.float16,
                                                reduce_dtype=torch.float16,
                                                buffer_dtype=torch.float16))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    # model.gradient_checkpointing_enable()
    model.train()

    def forward(data):
        return model(**data)

    def backward(loss):
        loss.backward()

    def optim():
        optimizer.step()
        optimizer.zero_grad()

    return forward, backward, optim, model_size


if __name__ == '__main__':
    import colossalai
    colossalai.launch_from_torch(config={})
    print(train_init('opt-1b'))
