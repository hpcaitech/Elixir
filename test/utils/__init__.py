from . import mlp, resnet, small
from .registry import TEST_MODELS


def assert_dict_keys(test_dict, keys):
    assert len(test_dict) == len(keys)
    for k in keys:
        assert k in test_dict
