import inspect

class Registry(object):
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_func, module_name=None, force=False):
        if not inspect.isfunction(module_func):
            raise TypeError(f'module must be a function, but got {type(module_func)}')

        if module_name is None:
            module_name = module_func.__name__
        assert isinstance(module_name, str)
        module_name = [module_name]

        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered in {self.name}')
            self._module_dict[name] = module_func

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the function name or the specified name, 
        and value is the function itself. It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> def resnet18():
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> def resnet18():
            >>>     pass
            >>> backbones.register_module(resnet18)
        Args:
            name (str | None): The module name to be registered. If not specified, the function name will be used.
            force (bool, optional): Whether to override an existing class with the same name. Default: False.
            module (type): Module to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_func=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        # use it as a decorator: @x.register_module()
        def _register(func):
            self._register_module(module_func=func, module_name=name, force=force)
            return func

        return _register