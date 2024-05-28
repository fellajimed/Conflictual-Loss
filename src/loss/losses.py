import torch
from torch import nn
import numpy as np
import sys
from random import randint

epsilon = sys.float_info.min

REG_TYPE_MAP = dict(
    LS='Label Smoothing',
    CP='Confidence Penalty',
    CL='Conflictual Loss',
)


class SoftplusCrossEntropyLoss(nn.Module):
    def __init__(self, beta=1, threshold=20, reduction='mean'):
        from ..metrics.uncertainties import SoftplusNormalization

        super().__init__()

        self.softplus = SoftplusNormalization(dim=1, beta=beta,
                                              threshold=threshold)
        self.reduction = reduction

    def extra_repr(self):
        return (f'reduction={self.reduction}')

    def forward(self, logits, targets):
        assert len(targets.shape) == 1 and len(logits.shape) == 2

        nll = - self.softplus(logits).gather(1, targets[:, None]).log()
        if self.reduction.lower() == 'mean':
            return nll.mean()
        elif self.reduction.lower() == 'sum':
            return nll.sum()
        else:
            return nll


class LossWithOutputDistReg(nn.Module):
    def __new__(cls, reg_type=None, ce_params=dict(),
                reg_params=dict(), **kwargs):
        if reg_type is None or reg_type not in REG_TYPE_MAP.keys():
            return nn.CrossEntropyLoss(**ce_params)
        return object.__new__(cls)

    def __init__(self, reg_type=None, ce_params=dict(),
                 reg_params=dict(), **kwargs):
        """
        LossWithOutputDistReg: Loss with outputs distributions regularization

        reg_type: (string) LS, CP, CR, DCR
        ce_params: dict of the parameters for the Cross Entropy Loss
        reg_params: dict of the parameters for the regularization
                    commun keys:
                        * `reg_coef` (float)
                        * `reduction` (string: 'sum' or 'mean')
                        * `class_index` (int) for the class regularization
        """
        super().__init__()

        self.reg_type = reg_type.upper()
        self.reg_type_map = REG_TYPE_MAP
        self.ce_params = ce_params
        self.reg_params = reg_params
        self.ce_loss = nn.CrossEntropyLoss(**ce_params)
        self.ce_weight = 1.

        self.reg_fct = getattr(self,
                               self.reg_type_map.get(self.reg_type)
                               .lower().replace(' ', '_'))(**reg_params)

        self.loss = self.loss_function()

    def loss_function(self):
        if self.reg_type == 'EDLL':
            def _fct(logits, targets, **kwargs):
                return self.reg_fct(logits, targets)
        else:
            def _fct(logits, targets, **kwargs):
                return (self.ce_weight * self.ce_loss(logits, targets)
                        + self.reg_fct(logits, **kwargs))
        return _fct

    def label_smoothing(self, reg_coef=1, reduction='mean', **kwargs):
        """
        Label Smoothing
        """
        def _reg_fct(logits, **kwargs):
            """
            logits torch.Tensor [BS, number of classes]

            return mean/sum of reg_coef * KL(unif(number of classes) || p)
            """
            probs = nn.functional.softmax(logits, dim=1).to(torch.double)
            reg_loss = - (np.log(logits.shape[-1])
                          + (torch.log(probs + epsilon)
                             ).sum(dim=-1) / np.log(logits.shape[-1]))
            if reduction == 'sum':
                return reg_coef * reg_loss.sum()
            else:
                # default: mean
                return reg_coef * reg_loss.mean()
        return _reg_fct

    def confidence_penalty(self, reg_coef=1, reduction='mean', **kwargs):
        """
        Confidence Penalty
        """
        def _reg_fct(logits, **kwargs):
            """
            logits torch.Tensor [BS, number of classes]

            return mean/sum of reg_coef * KL(p || unif(number of classes))
            """
            probs = nn.functional.softmax(logits, dim=1).to(torch.double)
            reg_loss = (np.log(logits.shape[-1])
                        + (probs * torch.log(probs + epsilon)).sum(dim=-1))
            if reduction == 'sum':
                return reg_coef * reg_loss.sum()
            else:
                # default: mean
                return reg_coef * reg_loss.mean()
        return _reg_fct

    def conflictual_loss(self, reg_coef=1, reduction='mean',
                         class_index=0, alpha=None,
                         len_train_set=None, **kwargs):
        '''
        Conflictual Loss

        either use the parameter `reg_coef` for the regularization coeffient
        or compute it based on `alpha` and `len_train_set`
        '''
        if alpha is not None and len_train_set is not None:
            reg_coef = alpha / len_train_set
            self.reg_params['reg_coef'] = reg_coef

        def index_gen(n):
            return randint(0, n-1) if class_index == -1 else class_index

        def _reg_fct(logits, class_index=None, **kwargs):
            """
            logits torch.Tensor [BS, number of classes]

            return mean/sum of class regularization
            """
            if class_index is None:
                class_index = index_gen(logits.shape[-1])

            log_probs = nn.functional.log_softmax(logits, dim=1,
                                                  dtype=torch.double)
            reg_loss = - log_probs[:, class_index]
            if reduction == 'sum':
                return reg_coef * reg_loss.sum()
            else:
                # default: mean
                return reg_coef * reg_loss.mean()
        return _reg_fct

    def extra_repr(self):
        if self.reg_fct is not None:
            return (f'(Regularization) {self.reg_type_map[self.reg_type]}: '
                    f'{self.reg_params}')

    def forward(self, logits, targets, **kwargs):
        return self.loss(logits, targets, **kwargs)
