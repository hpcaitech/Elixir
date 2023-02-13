def test_registry():
    from test.utils.registry import TEST_MODELS
    builder, train_iter, valid_iter, criterion = TEST_MODELS.get_func("mlp")()
    model = builder()
    data, label = next(train_iter)
    out = model(data)
    loss = criterion(out, label)
    loss.backward()
