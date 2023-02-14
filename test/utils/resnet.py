from test.utils.iterator import TestIterator
from test.utils.registry import TEST_MODELS

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetIterator(TestIterator):

    def generate(self):
        data = torch.randn(4, 3, 32, 32)
        label = torch.randint(low=0, high=10, size=(4,))
        return data, label


@TEST_MODELS.register('resnet')
def resnet_funcs():

    def model_builder():
        return resnet18()

    train_iter = ResNetIterator()
    valid_iter = ResNetIterator()

    criterion = nn.CrossEntropyLoss()

    return model_builder, train_iter, valid_iter, criterion
