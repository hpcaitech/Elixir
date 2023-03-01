from test.parameter.hf_models import test_hf_model

import torch
import transformers

BS = 2
SL = 16

one_sentence_config = transformers.AlbertConfig(embedding_size=128,
                                                hidden_size=128,
                                                num_hidden_layers=2,
                                                num_attention_heads=4,
                                                intermediate_size=256)

multi_sentence_config = transformers.AlbertConfig(hidden_size=128,
                                                  num_hidden_layers=2,
                                                  num_attention_heads=4,
                                                  intermediate_size=256)


def data_one_sentence():
    input_ids = torch.zeros((BS, SL), dtype=torch.int64)
    token_type_ids = torch.zeros((BS, SL), dtype=torch.int64)
    attention_mask = torch.zeros((BS, SL), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def data_qa():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    question, text = 'Who was Jim Henson?', 'Jim Henson was a nice puppet'
    inputs = tokenizer(question, text, return_tensors='pt')
    return inputs


def data_mcq():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    prompt = 'In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.'
    choice0 = 'It is eaten with a fork and a knife.'
    choice1 = 'It is eaten while held in the hand.'
    encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='pt', padding=True)
    encoding = {k: v.unsqueeze(0) for k, v in encoding.items()}
    return encoding


model_dict = {
    transformers.AlbertModel: dict(config=one_sentence_config, data=data_one_sentence),
    transformers.AlbertForPreTraining: dict(config=one_sentence_config, data=data_one_sentence),
    transformers.AlbertForMaskedLM: dict(config=one_sentence_config, data=data_one_sentence),
    transformers.AlbertForSequenceClassification: dict(config=one_sentence_config, data=data_one_sentence),
    transformers.AlbertForTokenClassification: dict(config=one_sentence_config, data=data_one_sentence),
    transformers.AlbertForQuestionAnswering: dict(config=multi_sentence_config, data=data_qa),
    transformers.AlbertForMultipleChoice: dict(config=multi_sentence_config, data=data_mcq),
}


def test_albert():
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
    test_albert()
