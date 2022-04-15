from ..utils.registry import Registry

backbone = Registry('backbone')


def build_classifier(name, **kwargs):
    if name not in backbone:
        raise KeyError(f'{name} is not in the {backbone.name} registry')
    net = backbone.get(name)
    return net(**kwargs)