import torch

from . import gpt, mlp, resnet, small
from .registry import TEST_MODELS


def assert_dict_keys(test_dict, keys):
    assert len(test_dict) == len(keys)
    for k in keys:
        assert k in test_dict


def assert_dict_values(da, db, fn):
    assert len(da) == len(db)
    for k, v in da.items():
        assert k in db
        if not torch.is_tensor(v):
            continue
        u = db.get(k)
        if u.device != v.device:
            v = v.to(u.device)
        assert fn(u.data, v.data)
