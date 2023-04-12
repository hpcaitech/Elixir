from test.utils.gpt import GPTLMModel

from transformers import AutoConfig, OPTConfig

from example.common.opt import OPTLMModel


def gpt2_400m():
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16)


def gpt2_1b():
    return GPTLMModel(hidden_size=1536, num_layers=32, num_attention_heads=16)


def gpt2_4b():
    return GPTLMModel(hidden_size=3072, num_layers=32, num_attention_heads=16)


def gpt2_10b():
    return GPTLMModel(hidden_size=4096, num_layers=48, num_attention_heads=16)


def gpt2_20b():
    return GPTLMModel(hidden_size=8192, num_layers=24, num_attention_heads=16)


def gpt2_25b():
    return GPTLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16)


def gpt2_30b():
    return GPTLMModel(hidden_size=8192, num_layers=36, num_attention_heads=16)


def gpt2_40b():
    return GPTLMModel(hidden_size=8192, num_layers=50, num_attention_heads=16)


def opt_350m():
    opt_config = AutoConfig.from_pretrained('facebook/opt-350m')
    return OPTLMModel(opt_config)


def opt_1b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    return OPTLMModel(opt_config)


def opt_3b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-2.7b')
    return OPTLMModel(opt_config)


def opt_7b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-6.7b')
    return OPTLMModel(opt_config)


def opt_13b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-13b')
    return OPTLMModel(opt_config)


def opt_30b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-30b')
    return OPTLMModel(opt_config)


def opt_66b():
    opt_config = AutoConfig.from_pretrained('facebook/opt-66b')
    return OPTLMModel(opt_config)


def opt_175b():
    opt_config = OPTConfig(activation_dropout=0.0,
                           hidden_size=12288,
                           num_hidden_layers=96,
                           ffn_dim=49152,
                           num_attention_heads=96,
                           word_embed_proj_dim=12288,
                           output_projection=True)
    return OPTLMModel(opt_config)


def get_model(name: str):
    if name == 'gpt2-400m':
        return gpt2_400m()
    elif name == 'gpt2-1b':
        return gpt2_1b()
    elif name == 'gpt2-4b':
        return gpt2_4b()
    elif name == 'gpt2-10b':
        return gpt2_10b()
    elif name == 'gpt2-20b':
        return gpt2_20b()
    elif name == 'gpt2-25b':
        return gpt2_25b()
    elif name == 'gpt2-30b':
        return gpt2_30b()
    elif name == 'gpt2-40b':
        return gpt2_40b()
    elif name == 'opt-350m':
        return opt_350m()
    elif name == 'opt-1b':
        return opt_1b()
    elif name == 'opt-3b':
        return opt_3b()
    elif name == 'opt-7b':
        return opt_7b()
    elif name == 'opt-13b':
        return opt_13b()
    elif name == 'opt-30b':
        return opt_30b()
    elif name == 'opt-66b':
        return opt_66b()
    elif name == 'opt-175b':
        return opt_175b()
    else:
        raise ValueError(f'Unknown model name: {name}')
