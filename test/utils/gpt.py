from test.utils.iterator import TestIterator
from test.utils.registry import TEST_MODELS

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

MICRO_VS = 128
MICRO_BS = 4
MICRO_SL = 64

MACRO_VS = 50257
MACRO_BS = 2
MACRO_SL = 1024


class MicroIterator(TestIterator):

    def generate(self):
        input_ids = torch.randint(low=0, high=MICRO_VS, size=(MICRO_BS, MICRO_SL))
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask


class MacroIterator(TestIterator):

    def generate(self):
        input_ids = torch.randint(low=0, hight=MACRO_VS, size=(MACRO_BS, MACRO_SL))
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class GPTLMModel(nn.Module):

    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257):
        super().__init__()
        self.enable_gc = False
        self.module = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size,
                       resid_pdrop=0.0,
                       embd_pdrop=0.0,
                       attn_pdrop=0.0))
        self.criterion = GPTLMLoss()

    def gradient_checkpointing_enable(self):
        self.module.gradient_checkpointing_enable()
        self.enable_gc = True

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        output = self.module(input_ids=input_ids, attention_mask=attention_mask, use_cache=(not self.enable_gc))[0]
        loss = self.criterion(output, input_ids)
        return loss


def gpt2_micro():
    return GPTLMModel(hidden_size=32, num_layers=2, num_attention_heads=4, max_seq_len=64, vocab_size=128)


def gpt2_small():
    return GPTLMModel()


def gpt2_base():
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16)


@TEST_MODELS.register('gpt_micro')
def gpt_micro_funcs():

    def model_builder():
        return gpt2_micro()

    train_iter = MicroIterator()
    valid_iter = MicroIterator()

    criterion = nn.CrossEntropyLoss()

    return model_builder, train_iter, valid_iter, criterion


@TEST_MODELS.register('gpt_small')
def gpt_base_funcs():

    def model_builder():
        return gpt2_small()

    train_iter = MacroIterator()
    valid_iter = MacroIterator()

    criterion = nn.CrossEntropyLoss()

    return model_builder, train_iter, valid_iter, criterion
