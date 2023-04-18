from elixir.ctx import MetaContext
from example.common.models import get_model

gpt_list = ['gpt2-400m', 'gpt2-1b']
opt_list = ['opt-350m', 'opt-1b', 'opt-3b', 'opt-7b', 'opt-13b', 'opt-30b', 'opt-66b', 'opt-175b']


def init_model(name: str):
    with MetaContext():
        model = get_model(name)
    del model


def main():
    for name in gpt_list:
        init_model(name)

    for name in opt_list:
        init_model(name)


if __name__ == '__main__':
    main()
