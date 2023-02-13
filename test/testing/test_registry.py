def test_registry():
    from test.utils.registry import TEST_MODELS
    for name, construct_func in TEST_MODELS:
        print(f"model `{name}` is in testing")
        builder, train_iter, valid_iter, criterion = construct_func()
        model = builder()
        data, label = next(train_iter)
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
