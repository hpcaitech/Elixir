from collections import OrderedDict


class Registry(object):

    def __init__(self) -> None:
        super().__init__()
        self._registry_dict = OrderedDict()

    def register(self, name: str):
        assert name not in self._registry_dict

        def register_func(call_func):
            self._registry_dict[name] = call_func
            return call_func

        return register_func

    def get_func(self, name: str):
        return self._registry_dict[name]

    def __iter__(self):
        return iter(self._registry_dict.items())


TEST_MODELS = Registry()

__all__ = [TEST_MODELS]
