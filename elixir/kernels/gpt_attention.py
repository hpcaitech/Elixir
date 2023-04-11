import einops
import torch
import xformers.ops as xops
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model


class XGPT2Attention(GPT2Attention):

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        assert self.scale_attn_weights
        assert not self.is_cross_attention
        assert not self.scale_attn_by_inverse_layer_idx
        assert not self.reorder_and_upcast_attn

        b_size, h_size, m_size, k_size = query.size()

        assert self.bias.size(-1) == m_size
        query = einops.rearrange(query, 'b h m k -> b m h k')
        key = einops.rearrange(key, 'b h m k -> b m h k')
        value = einops.rearrange(value, 'b h m k -> b m h k')

        drop_rate = self.attn_dropout.p
        output = xops.memory_efficient_attention(query, key, value, p=drop_rate, attn_bias=xops.LowerTriangularMask())

        ret = einops.rearrange(output, 'b m h k -> b h m k')

        return ret, None


class XGPT2Model(GPT2Model):

    def forward(self, *args, **kwargs):
        assert 'attention_mask' in kwargs, 'please pass attention_mask as a kwarg'
        attn_mask = kwargs.get('attention_mask')
        assert torch.all(attn_mask == 1), 'only accept no padding mask'

        head_mask = kwargs.get('head_mask', None)
        assert head_mask is None, 'head mask should be None'

        output_attn = kwargs.get('output_attentions', False)
        assert output_attn is False or output_attn is None

        return super().forward(*args, **kwargs)
