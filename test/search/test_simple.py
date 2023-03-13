from copy import deepcopy
from test.utils import TEST_MODELS

from elixir.search import simple_search


def test_simple_search():
    builder, *_ = TEST_MODELS.get_func('small')()
    model = builder()
    sr = simple_search(model, 1, split_number=5)

    config_list = deepcopy(sr.chunk_config_list)

    private_dict = config_list.pop(0)
    assert private_dict['name_list'] == ['embed.weight']
    assert private_dict['chunk_size'] == 320

    assert config_list[0]['name_list'] == ['norm1.weight', 'norm1.bias']
    assert config_list[1]['name_list'] == ['mlp.proj1.weight', 'mlp.proj1.bias']
    assert config_list[2]['name_list'] == ['mlp.proj2.weight', 'mlp.proj2.bias']
    assert config_list[3]['name_list'] == ['norm2.weight']
    assert config_list[4]['name_list'] == ['norm2.bias']

    for config_dict in config_list:
        assert config_dict['chunk_size'] == 1088


if __name__ == '__main__':
    test_simple_search()
