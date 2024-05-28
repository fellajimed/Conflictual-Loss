"""
code inspired by: https://github.com/danruod/IEDL/tree/main
"""
import torch
from torch import nn

from ..models import ToProbs


EVIDENTIAL_LOSS_MAP = dict(
    EDL='Evidential Deep Learning',
    IEDL='Fisher Information-based Evidential Deep Learning'
)

ACTIVATION_FCT = dict(relu=nn.ReLU(), softplus=nn.Softplus(), exp=torch.exp)


def compute_alpha_hat(alphas, targets, kl_ref_coef):
    if alphas.shape != targets.shape:
        targets = torch.zeros_like(alphas).scatter_(
            -1, targets.unsqueeze(-1), 1)
    return alphas * (1 - targets) + kl_ref_coef * targets


class EvidentialLoss(nn.Module):
    # TODO: change default value of `add_constant` to `0.`
    def __init__(self, loss_type, kl_coef=None, ref_coef=1.,
                 fisher_coef=None, activation_fct='softplus',
                 add_constant=1., epoch_update=False, **kwargs):
        super().__init__()

        self.loss_type = loss_type.upper()
        if self.loss_type not in EVIDENTIAL_LOSS_MAP.keys():
            raise NotImplementedError(
                f"loss_type not valid. Given {loss_type}."
                f" Expected value in {list(EVIDENTIAL_LOSS_MAP.keys())}")

        self.add_constant = add_constant
        self.kl_coef = kl_coef
        assert ref_coef > 0
        self.kl_ref_coef = ref_coef
        self.epoch_update = epoch_update
        self.fisher_coef = fisher_coef

        if activation_fct.lower() not in ACTIVATION_FCT.keys():
            activation_fct = 'softplus'

        self.activation_fct_name = activation_fct
        self.activation_fct = ACTIVATION_FCT[activation_fct.lower()]

        # loss function
        self.loss = self.loss_function()

        # to probs kwargs
        self.toprobs_kwargs = dict(activation_fct=activation_fct.lower(),
                                   loss_type='edl',
                                   add_constant=add_constant)
        self.toprobs = ToProbs(**self.toprobs_kwargs)

    def loss_function(self):
        def _fct(logits, targets, reduction='mean', epoch=None):
            alphas = self.toprobs(logits, return_probs=False)
            if logits.shape != targets.shape:
                targets = torch.zeros_like(logits).scatter_(
                    -1, targets.unsqueeze(-1), 1)
            if all((self.epoch_update, epoch is not None,
                    self.kl_coef is None)):
                # This value is taken from the implementation of IEDL
                self.kl_coef = min(1., epoch/10.)
            return (self.mse_term()(alphas, targets, reduction)
                    + self.kl_term()(alphas, targets, reduction)
                    + self.fisher_information_term()(alphas, reduction))
        return _fct

    def mse_term(self):
        def _fct(alphas, targets_1hot, reduction='mean'):
            alphas_0 = torch.sum(alphas, dim=-1, keepdim=True)
            mse = (alphas * (alphas_0 - alphas)
                   / ((alphas_0 + 1) * alphas_0**2)
                   + (targets_1hot - alphas / alphas_0)**2)
            if self.loss_type == 'IEDL':
                mse *= torch.polygamma(1, alphas)
            mse = torch.sum(mse, dim=-1)
            return mse.mean() if reduction == 'mean' else mse.sum()
        return _fct

    def kl_term(self):
        r"""
        KL(Dir(x, \hat(\alpha)) || Dir(x, self.kl_ref_coef))
        """
        if self.kl_coef is None:
            return lambda x, y, z: 0

        def _fct(alphas, targets_1hot, reduction='mean'):
            nb_classes = alphas.shape[-1]
            alphas_hat = compute_alpha_hat(alphas, targets_1hot,
                                           self.kl_ref_coef)
            alphas_hat_0 = torch.sum(alphas_hat, dim=-1, keepdim=True)
            kl = (torch.lgamma(alphas_hat_0)
                  - torch.lgamma(torch.tensor(self.kl_ref_coef * nb_classes))
                  + nb_classes * torch.lgamma(torch.tensor(self.kl_ref_coef))
                  - torch.sum(torch.lgamma(alphas_hat), dim=-1, keepdim=True)
                  + torch.sum((alphas_hat - self.kl_ref_coef)
                              * (torch.digamma(alphas_hat)
                                 - torch.digamma(alphas_hat_0)),
                              dim=-1, keepdim=True))
            kl *= self.kl_coef
            return kl.mean() if reduction == 'mean' else kl.sum()
        return _fct

    def fisher_information_term(self):
        if self.fisher_coef is None or self.loss_type != 'IEDL':
            return lambda x, y: 0

        def _fct(alphas, reduction='mean'):
            alphas_0 = torch.sum(alphas, dim=-1, keepdim=True)
            _polyg_alpha = torch.polygamma(1, alphas)
            _polyg_alpha_0 = torch.polygamma(1, alphas_0)
            f_info = (torch.sum(torch.log(_polyg_alpha), dim=-1)
                      + torch.log(1 - torch.sum(_polyg_alpha_0
                                                / _polyg_alpha, dim=-1)))
            f_info *= - self.fisher_coef
            return f_info.mean() if reduction == 'mean' else f_info.sum()
        return _fct

    def extra_repr(self):
        kl_coef = 0. if self.kl_coef is None else self.kl_coef
        fisher_coef = 0. if self.fisher_coef is None else self.fisher_coef
        return (f"{EVIDENTIAL_LOSS_MAP[self.loss_type]}: ({kl_coef=:.2e}"
                f" (ref_coef={self.kl_ref_coef:.2e}) - {fisher_coef=:.2e} "
                f"- activation_fct={self.activation_fct_name})")

    def forward(self, logits, targets, reduction='mean', epoch=None):
        return self.loss(logits, targets, reduction, epoch)
