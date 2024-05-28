from .evidential import EvidentialLoss
from .losses import LossWithOutputDistReg, SoftplusCrossEntropyLoss


class Loss:
    def __new__(cls, reg_type=None, loss_type=None, beta=1,
                threshold=20, reduction='mean', **kwargs):
        if loss_type is not None and isinstance(loss_type, str):
            if 'softplus' in loss_type.lower():
                return SoftplusCrossEntropyLoss(beta=beta, threshold=threshold,
                                                reduction=reduction)
            else:
                return EvidentialLoss(loss_type=loss_type, **kwargs)
        else:
            return LossWithOutputDistReg(reg_type=reg_type, **kwargs)


__all__ = ['Loss', 'EvidentialLoss',
           'LossWithOutputDistReg', 'SoftplusCrossEntropyLoss']
