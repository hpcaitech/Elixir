from test.utils import TEST_MODELS

from elixir.ctx import MetaContext
from elixir.tracer.param_tracer import generate_fx_order


def main():
    builder, *_ = TEST_MODELS.get_func('resnet')()
    with MetaContext():
        model = builder()

    for name, param in model.named_parameters():
        print(name, param)

    for name, buffer in model.named_buffers():
        print(name, buffer)

    print(generate_fx_order(model))


if __name__ == '__main__':
    main()
