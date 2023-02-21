import torch

from elixir.ctx import tensor_creation_methods


class MetaContext(object):

    def __init__(self) -> None:
        super().__init__()
        return None

    def __enter__(self):

        def meta_wrap(func):

            def wrapped_func(*args, **kwargs):
                if 'device' in kwargs:
                    kwargs['device'] = 'meta'
                else:
                    kwargs.update({'device': 'meta'})
                return func(*args, **kwargs)

            return wrapped_func

        for name, method in tensor_creation_methods.items():
            setattr(torch, name, meta_wrap(method))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, method in tensor_creation_methods.items():
            setattr(torch, name, method)
