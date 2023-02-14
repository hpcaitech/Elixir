import contextlib
from collections import defaultdict
from typing import Dict, Iterator, NamedTuple

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils._pytree import tree_map


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


PARAM_INFO_DICT: Dict[int, str] = defaultdict(str)


class MyTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        # TODO: clone strides and storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad)
        r.elem = elem
        return r

    def __repr__(self):
        return f'MyTensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f'{func}:')

        def print_tensor(x):
            if isinstance(x, Parameter) or type(x) is torch.Tensor:
                grad_name = None
                if x.grad_fn:
                    grad_name = x.grad_fn
                    print(x.grad_fn.next_functions[0][0])
                print(id(x), x.shape, x.data_ptr(), type(x), grad_name)

        def print_param(x):
            if isinstance(x, torch.Tensor):
                data_ptr = x.data_ptr()
                if data_ptr in PARAM_INFO_DICT:
                    print(f'Get Parameter `{PARAM_INFO_DICT[data_ptr]}`')

        tree_map(print_param, args)
        tree_map(print_param, kwargs)

        def unwrap(x):
            return x.elem if isinstance(x, MyTensor) else x

        def wrap(x):
            return MyTensor(x) if isinstance(x, torch.Tensor) else x

        with no_dispatch():
            res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if len(res) == 1:
            return res[0]
        else:
            return res


def get_param_info(model: nn.Module):
    for name, param in model.named_parameters():
        PARAM_INFO_DICT[param.data_ptr()] = name


def run_linear():
    model = nn.Linear(4, 8)
    for param in model.parameters():
        param.to('meta')
    get_param_info(model)

    inp = MyTensor(torch.randn(6, 4, device='meta', requires_grad=True))
    model(inp).sum().backward()

    torch.cuda.synchronize()
    print('Done!')


def run_resnet():
    import torchvision.models as models
    model = models.resnet18().cuda()
    get_param_info(model)

    inp = MyTensor(torch.randn(1, 3, 224, 224, device='cuda', requires_grad=True))
    model(inp).sum().backward()

    torch.cuda.synchronize()
    print('Done!')


def run_bert():
    from transformers import BertConfig, BertForSequenceClassification
    config = BertConfig(vocab_size=16,
                        gradient_checkpointing=True,
                        hidden_size=8,
                        intermediate_size=8 * 4,
                        num_attention_heads=2,
                        max_position_embeddings=8,
                        num_hidden_layers=2,
                        hidden_dropout_prob=0.,
                        attention_probs_dropout_prob=0.)
    model = BertForSequenceClassification(config).cuda()
    model.gradient_checkpointing_enable()
    get_param_info(model)

    data = torch.randint(low=0, high=8, size=(4, 8), device='cuda', dtype=torch.long)
    data = MyTensor(data)
    model(data)[0].sum().backward()

    torch.cuda.synchronize()
    print('Done!')


if __name__ == '__main__':
    run_bert()
