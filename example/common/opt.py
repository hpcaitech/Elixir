import torch.nn as nn
from transformers import OPTForCausalLM


class OPTLMModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.module = OPTForCausalLM(config=config)
        self.enable_gc = False

    def gradient_checkpointing_enable(self):
        self.module.gradient_checkpointing_enable()
        self.enable_gc = True

    def forward(self, input_ids, attention_mask):
        loss = self.module(
        # pre-commit: do not rearrange
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=(not self.enable_gc))['loss']
        return loss
