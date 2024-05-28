import torch
from torch import nn
from itertools import tee

from .classification import ClassificationModuleNet
from .mlp import MLPNet


def pairwise(iterable):
    """
    pairwise('ABCDEFG') --> AB BC CD DE EF FG

    New in python 3.10: from itertools import pairwaise
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def conv_block(in_chan: int, out_chan: int):
    return nn.Sequential(nn.Conv2d(in_chan, out_chan, 3, padding=1),
                         nn.Conv2d(out_chan, out_chan, 3, padding=1),
                         nn.ReLU(), nn.MaxPool2d(2, stride=2))


class LeNetish(ClassificationModuleNet):
    def __init__(self, input_shape, out_channels, nb_classes,
                 device, **kwargs_mlp):
        super().__init__(input_shape, nb_classes, device)

        self.in_channels = input_shape[0] if len(input_shape) > 2 else 1

        self.channels = [self.in_channels] + list(out_channels)

        self.conv = nn.Sequential(
            *[conv_block(in_chan, out_chan)
                for (in_chan, out_chan) in pairwise(self.channels)],
            nn.Flatten())

        x = torch.rand((1, *self.input_shape))
        self.in_linear = self.conv(x).shape[-1]

        if kwargs_mlp:
            # changing the fully connnected layer (fc) with an MLP model
            self.linear = MLPNet(input_shape=self.in_linear,
                                 nb_classes=nb_classes,
                                 device=device, **kwargs_mlp)
        else:
            # if the kwargs of the MLP are empty, replace the fc with a Linear
            self.linear = nn.Linear(self.in_linear, nb_classes)

        self.to(self.device)

    def forward(self, x):
        return self.linear(self.conv(x))
