from test.parameter.hf_models import test_hf_model

import torch
import transformers

BS = 2
SL = 16

gpt_config = transformers.GPT2Config(n_positions=64, n_embd=128, n_layer=2, n_head=4, pad_token_id=0)
opt_config = transformers.OPTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, pad_token_id=0)
t5_config = transformers.T5Config(d_model=128, d_kv=32, d_ff=256, num_layers=2, num_heads=4, pad_token_id=0)


def data_gpt():
    input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    token_type_ids = torch.zeros((BS, SL), dtype=torch.int64)
    attention_mask = torch.zeros((BS, SL), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def data_opt():
    input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    attention_mask = torch.zeros((BS, SL), dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_t5():
    input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    decoder_input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    return dict(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def data_t5_encoder():
    input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    return dict(input_ids=input_ids)


model_dict = {
    transformers.GPT2Model: dict(config=gpt_config, data=data_gpt),
    transformers.GPT2LMHeadModel: dict(config=gpt_config, data=data_gpt),
    transformers.GPT2DoubleHeadsModel: dict(config=gpt_config, data=data_gpt),
    transformers.GPT2ForTokenClassification: dict(config=gpt_config, data=data_gpt),
    transformers.GPT2ForSequenceClassification: dict(config=gpt_config, data=data_gpt),
    transformers.OPTModel: dict(config=opt_config, data=data_opt),
    transformers.OPTForCausalLM: dict(config=opt_config, data=data_opt),
    transformers.T5EncoderModel: dict(config=t5_config, data=data_t5_encoder),
    transformers.T5Model: dict(config=t5_config, data=data_t5),
    transformers.T5ForConditionalGeneration: dict(config=t5_config, data=data_t5),
}


def test_gpt():
    for builder, config_dict in model_dict.items():
        kwargs = dict(config=config_dict['config'])
        data_fn = config_dict['data']

        flag = '√'
        try:
            test_hf_model(builder, kwargs, data_fn)
        except:
            flag = 'x'
        print(f'{builder.__name__:40s} {flag}')


if __name__ == '__main__':
    test_gpt()
