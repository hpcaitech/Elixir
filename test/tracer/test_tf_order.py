from test.utils import TEST_MODELS

from elixir.tracer.param_tracer import generate_tf_order


def test_tf_forward_backward():
    builder, train_iter, test_iter, criterion = TEST_MODELS.get_func('gpt_micro')()
    model = builder()
    input_ids, attn_mask = next(train_iter)
    inp = dict(input_ids=input_ids, attention_mask=attn_mask)

    def forward_backward_fn(model, input_dict):
        model(**input_dict).sum().backward()

    # model.gradient_checkpointing_enable()
    tf_order = generate_tf_order(model, inp, forward_backward_fn)
    assert len(tf_order) == 32

    model.gradient_checkpointing_enable()
    tf_order = generate_tf_order(model, inp, forward_backward_fn)
    assert len(tf_order) == 44

    assert input_ids.device.type == 'cpu'
    assert attn_mask.device.type == 'cpu'
    for param in model.parameters():
        assert param.device.type == 'cpu'


if __name__ == '__main__':
    test_tf_forward_backward()
