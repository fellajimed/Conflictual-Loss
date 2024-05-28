"""
EXPERIMENTAL: see examples below
"""
import numpy as np
from torch import nn

from .classification import ClassificationModuleNet


class CustomSeqModel(ClassificationModuleNet):
    def __init__(self, input_shape, nb_classes, device='cpu', layers=None):
        """
        layer is an iterable of dict:
        * key (str): layer name - case sensitive !!
        * value: dict(args=Iterable, kwargs=dict())
        """
        from .. import models as pydbdl_models

        super().__init__(input_shape, nb_classes, device)

        if layers is None:
            layers = [dict(Linear=dict(args=[np.prod(input_shape), nb_classes],
                                       kwargs=dict()))]

        self.layers = []
        for layer in layers:
            assert len(layer) == 1
            key = list(layer.keys())[0]
            params = layer[key]
            try:
                layer = getattr(nn, key)
            except AttributeError:
                try:
                    layer = getattr(pydbdl_models, key)
                except AttributeError:
                    raise AttributeError("Could not initiate CustomSeqModel."
                                         f" {key} is not a valid layer name")

            self.layers.append(layer(*params.get('args', ()),
                                     **params.get('kwargs', {})))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch

    x = torch.randn(18, 3, 50)

    layers = [
        dict(Flatten=dict()),
        dict(Linear=dict(args=(150, 50))),
        dict(Linear=dict(kwargs=dict(in_features=50, out_features=10))),
    ]

    model = CustomSeqModel(input_shape=(3, 50), nb_classes=10, layers=layers)
    print(model)

    print(f"{model(x).shape=}")
    print("-"*70)
    layers = [
        dict(MLPNet=dict(kwargs=dict(input_shape=(3, 15), nb_classes=50,
                                     hidden_layers=[200, 100],
                                     dropout_rates=0.5))),
        dict(Linear=dict(kwargs=dict(in_features=50, out_features=10))),
    ]

    model = CustomSeqModel(input_shape=(3, 50), nb_classes=10, layers=layers)
    print(model)

    print(f"{model(x).shape=}")
