import torch
import torch.nn as nn
from torchvision.models import resnet18

from elixir.parameter.temp import MyParameter


class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(4, 16)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(16, 4)

    def forward(self, x):
        return x + self.proj2(self.act(self.proj1(x)))


def transform_module(model: nn.Module):
    for m_name, module in model.named_modules():
        param_list = list(module.named_parameters(recurse=False))
        for p_name, param in param_list:
            new_param = MyParameter(param.data)
            delattr(module, p_name)
            setattr(module, p_name, new_param)


def main():
    m = resnet18().cuda()
    for module in m.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    transform_module(m)
    print('transform: ok')

    # for name, param in m.named_parameters():
    #     print(name, type(param), param.shape, param.requires_grad)

    # d = torch.randn(4, 4, device='cuda')
    d = torch.randn(4, 3, 64, 64, device='cuda')
    m(d).sum()

    exit(0)


if __name__ == '__main__':
    main()
