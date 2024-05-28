from collections.abc import Iterable

import torch
from torch import nn
import numpy as np

from .classification import ClassificationModuleNet

# logger object
import logging
logger = logging.getLogger('main_all')


class MLPNet(ClassificationModuleNet):
    def __init__(self,
                 input_shape,
                 nb_classes,
                 hidden_layers=None,
                 dropout_rates=None,
                 device=torch.device('cpu'),
                 _relu=True,
                 lin_SN=False,
                 dropout_mask=False,
                 **kwargs):
        '''
        input_shape: could be a scaler or a tuple or a list
        nb_classes: (int) the number of the classes for the output
        hidden_layers: list of the number of units per hidden layer
                       - the length of the list is the number of hidden layers
                       - could be an iterable or a number
        dropout_rates: list of the values for the dropout rates
                       - could be an iterable or a number
                       - in the case of an iterable, the length is either equal
                       to len(hidden_layers) or len(hidden_layers)+1. In the
                       latter, the dropout is also applied to the input layer
        device: gpu or cpu
        _relu: (bool) apply relu activation function after each linear layer
        '''
        super().__init__(input_shape, nb_classes, device)

        self.use_relu = _relu

        # hidden layers params
        if not isinstance(hidden_layers, Iterable):
            if hidden_layers is not None:
                self.hidden_dims = [int(hidden_layers)]
            else:
                self.hidden_dims = []
        else:
            self.hidden_dims = list(hidden_layers)

        # dropout params
        if not isinstance(dropout_rates, Iterable):
            if dropout_rates is not None:
                self.dropout_rates = [float(dropout_rates)]*len(hidden_layers)
            else:
                self.dropout_rates = []
        else:
            self.dropout_rates = list(dropout_rates)

        self.use_dropout = False
        self.use_dropout_for_input = False
        if len(self.dropout_rates) > 0:
            if len(self.dropout_rates) == len(self.hidden_dims):
                self.use_dropout = True
            elif len(self.dropout_rates) == len(self.hidden_dims) + 1:
                self.use_dropout_for_input = True
                self.input_dropout = self.dropout_rates.pop(0)
                self.use_dropout = True
                logger.info("Dropout will be added to the input layer")
            else:
                logger.warning("The lengths of the list dropout_rates and the"
                               " list hidden_layers do not match: len(dropout"
                               f"_rates)={len(self.dropout_rates)} while "
                               f"len(hidden_layers)={len(self.hidden_dims)}"
                               "\n Warning: Dropout will not be used!")

        # spectral normalization
        self.lin_SN = lin_SN

        # create the classifier
        self.input_dim = np.prod(input_shape)
        self._create_classifier()

        # model self to device
        self.to(self.device)

    def _create_classifier(self):
        from .utils import apply_SN

        self.linear_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        self.all_linear_layers = [apply_SN(nn.Linear(in_dim, out_dim),
                                           self.lin_SN)
                                  for (in_dim, out_dim) in zip(
                                      self.linear_dims[0:-1],
                                      self.linear_dims[1:])]
        if self.use_dropout:
            self.dropout_layers = [nn.Dropout(p=rate)
                                   for rate in self.dropout_rates]
        else:
            self.dropout_layers = [None
                                   for _ in range(len(self.hidden_dims))]

        # define classifier
        self.classifier = []

        # apply dropout to input vector
        if self.use_dropout_for_input:
            self.classifier.append(nn.Dropout(p=self.input_dropout))

        # create blocks: Linear-Dropout-ReLU
        for lin_layer, drop_layer in zip(self.all_linear_layers[:-1],
                                         self.dropout_layers):
            self.classifier.append(lin_layer)
            if self.use_dropout:
                self.classifier.append(drop_layer)
            if self.use_relu:
                self.classifier.append(nn.ReLU())

        self.classifier.append(self.all_linear_layers[-1])

        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        y = self.classifier(x)
        return y
