import torch
from torch import nn


class Normalization(nn.Module):
    def __init__(self, positive_activation, add_constant=0.):
        super().__init__()
        self.activation = positive_activation
        self.add_const = add_constant

    def forward(self, x, return_probs=True):
        if self.activation == torch.exp and return_probs:
            return nn.Softmax(dim=-1)(x + self.add_const)

        x = self.activation(x) + self.add_const
        if return_probs:
            return x / x.sum(dim=-1, keepdim=True)
        else:
            return x


class ToProbs(nn.Module):
    def __init__(self, activation_fct=None, loss_type=None,
                 add_constant=0., **kwargs):
        super().__init__()
        if activation_fct is None or not isinstance(activation_fct, str):
            activation_fct = 'softmax'
        self.activation_fct = activation_fct.lower()

        self.make_probs_layer(loss_type, add_constant, **kwargs)

    def make_probs_layer(self, loss_type=None, add_constant=0., **kwargs):
        not_edl = (loss_type is None or
                   (isinstance(loss_type, str) and
                    'edl' not in loss_type.lower()))
        if not_edl:
            if self.activation_fct == 'softmax':
                self.probs_layer = Normalization(torch.exp)
            elif self.activation_fct == 'softplus':
                self.probs_layer = Normalization(nn.Softplus())
            else:
                # default
                self.probs_layer = Normalization(torch.exp)
        else:
            # Only valid for positive activation functions
            if self.activation_fct == 'softplus':
                self.probs_layer = Normalization(nn.Softplus(), add_constant)
            elif self.activation_fct == 'relu':
                self.probs_layer = Normalization(nn.ReLU(), add_constant)
            elif self.activation_fct == 'exp':
                self.probs_layer = Normalization(torch.exp, add_constant)
            else:
                # default
                self.probs_layer = Normalization(nn.Softplus(), add_constant)

    def forward(self, x, return_probs=True):
        if isinstance(self.probs_layer, Normalization):
            return self.probs_layer(x, return_probs)
        else:
            return self.probs_layer(x)


if __name__ == "__main__":
    data = torch.randn(5, 20)
    model = nn.Linear(20, 10)
    outputs = model(data)

    for act in ('softmax', 'softplus'):
        probs_layer = ToProbs(activation_fct=act, loss_type=None)
        probs = probs_layer(outputs)
        sum_probs = probs.sum(dim=1)
        assert torch.all(torch.isclose(sum_probs, torch.ones_like(sum_probs)))

    for (i, act) in enumerate(('softplus', 'relu', 'exp')):
        probs_layer = ToProbs(activation_fct=act, loss_type='EDL',
                              add_constant=i)
        probs = probs_layer(outputs)
        sum_probs = probs.sum(dim=1)
        assert torch.all(torch.isclose(sum_probs, torch.ones_like(sum_probs)))

    print('Tests passed with success!')
