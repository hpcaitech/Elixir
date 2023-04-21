import torch
import torch.distributed as dist
from colossalai.nn.optimizer import HybridAdam
from transformers.modeling_utils import no_init_weights

from elixir.ctx import MetaContext
from elixir.kernels.attn_wrapper import wrap_attention
from elixir.search import optimal_search
from elixir.utils import get_model_size
from elixir.wrapper import ElixirModule, ElixirOptimizer
from example.common.models import get_model


def train_step(model, data):
    loss = model(**data)
    loss.backward()
    return loss


def train_init(model_name: str, data: dict):
    global_group = dist.GroupMember.WORLD
    global_size = dist.get_world_size()

    with no_init_weights():
        model = get_model(model_name)
    model_size = get_model_size(model)
    optimizer = HybridAdam(model.parameters(), lr=1e-3)

    model.gradient_checkpointing_enable()
    model = wrap_attention(model)
    
    sr = optimal_search(model,
                        global_size,
                        unified_dtype=torch.float16,
                        overlap=True,
                        verbose=True,
                        inp=data,
                        step_fn=train_step)
    model = ElixirModule(model, sr, global_group, prefetch=True, dtype=torch.float16, use_fused_kernels=True)
    optimizer = ElixirOptimizer(model, optimizer, initial_scale=2048)

    model.train()

    def forward(data):
        return model(**data)

    def backward(loss):
        optimizer.backward(loss)

    def optim():
        optimizer.step()
        optimizer.zero_grad()

    return forward, backward, optim, model_size


if __name__ == '__main__':
    import colossalai
    colossalai.launch_from_torch(config={})
    print(train_init('opt-13b'))
