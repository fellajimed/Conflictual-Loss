import sys
import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
from functools import partial


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time
    model is torch model"""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model


def check_dropout(model):
    """
    check if a model has Dropout layer
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            return True
    return False


class SoftplusNormalization(nn.Module):
    def __init__(self, dim, beta=1, threshold=20):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)
        self.dim = dim

    def forward(self, x):
        y = self.softplus(x)
        return y / y.sum(dim=self.dim, keepdim=True)


def compute_uncertainties(stacked_predictions, backend='numpy'):
    """
    compute uncertainties from stacked predictions
    stacked_predictions: torch of shape (samples, n, classes) with n is the
                         number of forward passes (or of ensemble models)
    backend: numpy or torch
    """
    if backend not in ['numpy', 'torch']:
        raise ValueError("Expected backend to be either `numpy` or"
                         f" `torch`. Given {backend}")

    np_backend = (backend == 'numpy')

    # compute mean
    if np_backend:
        mean = np.mean(stacked_predictions, axis=1)
    else:
        mean = stacked_predictions.mean(dim=1)

    # compute variance
    if np_backend:
        variance = np.var(stacked_predictions, axis=1)
    else:
        variance = stacked_predictions.var(dim=1)

    # avoid `-inf` when computing the log
    epsilon = sys.float_info.min

    # compute entropy (predictive uncertainty)
    if np_backend:
        entropy = -np.sum(mean * np.log2(mean + epsilon, where=mean > 0),
                          axis=-1)
    else:
        entropy = -(mean * torch.log2(mean + epsilon)).sum(dim=-1)

    # compute conditional entropy (aleatoric uncertainty)
    if np_backend:
        conditional_entropy = -np.mean(np.sum(
            stacked_predictions * np.log2(stacked_predictions + epsilon,
                                          where=stacked_predictions > 0),
            axis=-1), axis=1)
    else:
        conditional_entropy = -((stacked_predictions *
                                 torch.log2(stacked_predictions + epsilon)
                                 ).sum(dim=-1)).mean(dim=1)

    # compute mutual information (model/epistimic uncertainty)
    mutual_info = entropy - conditional_entropy

    metrics = [mean, variance, entropy, conditional_entropy, mutual_info]

    if np_backend:
        return metrics
    else:
        return [metric.cpu() for metric in metrics]


def MC_Dropout(model, nbr_forward_passes, loader, n_classes,
               use_softmax=True, backend='numpy'):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    model : nn.Module
        pytorch model
    nbr_forward_passes : int
        number of Monte-Carlo samples/forward passes
    loader : object
        data loader object from the data loader module
    n_classes : int
        number of classes in the dataset
    backend : str
        backend to be used to compute the uncertainties
        possbile values: numpy, torch
    """
    device = next(model.parameters()).device

    if backend not in ['numpy', 'torch']:
        raise ValueError("Expected backend to be either `numpy` or"
                         f" `torch`. Given {backend}")

    act_fct = (nn.Softmax(dim=1) if use_softmax
               else SoftplusNormalization(dim=1))

    # mean, variance, entropy, conditional entropy, mutual information
    uncertainties = [[] for _ in range(5)]
    with torch.no_grad():
        for data, _ in tqdm(loader, desc="MC-Dropout", leave=False):
            data = data.to(device)
            # shape: [BS, nbr_forward_passes, n classes]
            predictions = torch.stack([act_fct(model(data)).to(torch.double)
                                       for _ in range(nbr_forward_passes)],
                                      dim=1)
            if backend == 'numpy':
                predictions = predictions.cpu().detach().numpy()

            batch_uncer = compute_uncertainties(predictions, backend)
            for metric, batch_value in zip(uncertainties, batch_uncer):
                metric.append(batch_value)

    if backend == 'numpy':
        return [np.concatenate(metric, axis=0) for metric in uncertainties]
    else:
        return [torch.cat(metric, dim=0) for metric in uncertainties]


def ensemble_uncertainties(ensemble_model, loader, backend='torch',
                           use_softmax=True, mc_nbr_forward_passes=1):
    """
    ensemble_model: EnsembleNet object
    loader: torch dataloader
    backend: numpy or torch
    mc_nbr_forward_passes: (int) allow to combine ensemble uncertainties
                            and MC-Dropout
    """
    if backend not in ['numpy', 'torch']:
        raise ValueError("Expected backend to be either `numpy` or"
                         f" `torch`. Given {backend}")

    device = next(ensemble_model.parameters()).device

    # for the ensemble models, dim==2 is the output dimension
    act_fct = (nn.Softmax(dim=2) if use_softmax
               else SoftplusNormalization(dim=2))

    if mc_nbr_forward_passes > 1:
        # check if the model has a Dropout layer
        apply_mc = check_dropout(ensemble_model)
    else:
        apply_mc = False

    # mean, variance, entropy, conditional entropy, mutual information
    uncertainties = [[] for _ in range(5)]
    with torch.no_grad():
        ensemble_model.eval()
        for data, _ in tqdm(loader, desc='Ensemble uncertainties',
                            leave=False):
            data = data.to(device)
            # add gaussian noise
            # data += torch.randn_like(data) * 100.
            logits = []
            for _ in range(mc_nbr_forward_passes):
                if apply_mc:
                    enable_dropout(ensemble_model)
                logits.append(ensemble_model(data, return_mean=False))
            logits = torch.cat(logits, dim=1)
            predictions = act_fct(logits).to(torch.double)
            if backend == 'numpy':
                predictions = predictions.cpu().detach().numpy()

            batch_uncer = compute_uncertainties(predictions, backend)
            for metric, batch_value in zip(uncertainties, batch_uncer):
                metric.append(batch_value)

    if backend == 'numpy':
        return [np.concatenate(metric, axis=0) for metric in uncertainties]
    else:
        return [torch.cat(metric, dim=0) for metric in uncertainties]


def expected_data_uncertainty(alphas):
    alphas_0 = torch.sum(alphas, dim=-1, keepdim=True)
    return torch.sum((torch.digamma(alphas_0 + 1) - torch.digamma(alphas + 1)
                      ) * alphas / alphas_0, dim=-1)


def total_uncertainty(alphas):
    ratio = alphas / torch.sum(alphas, dim=-1, keepdim=True)
    return - torch.sum(ratio * torch.log(ratio), dim=-1)


def distributional_uncertainty(alphas):
    return total_uncertainty(alphas) - expected_data_uncertainty(alphas)


def differential_entropy(alphas):
    """
    a common measure of distributional uncertainty
    """
    alphas_0 = torch.sum(alphas, dim=-1, keepdim=True)
    return (- torch.lgamma(alphas_0).squeeze() + torch.sum(
        torch.lgamma(alphas) - (alphas - 1) * (
            torch.digamma(alphas) - torch.digamma(alphas_0)), dim=-1))


def compute_evidential_uncertainties(alphas, backend='torch'):
    alphas = alphas.squeeze()

    uncer_fcts = [total_uncertainty, expected_data_uncertainty,
                  distributional_uncertainty, differential_entropy]
    values = [uncer_fct(alphas) for uncer_fct in uncer_fcts]
    if backend == 'numpy':
        return [value.cpu().detach().numpy() for value in values]
    else:
        return values


def evidential_uncertainties(model, loader, activation_fct='softplus',
                             loss_type='edl', add_constant=1.,
                             backend='torch', **kwargs):
    """
    Uncertainties for Dirichlet-based Uncertainty Models

    return a list of 4 arrays:
        * total uncertainty
        * expected data uncertainty
        * distributional uncertainty
        * differential entropy
    """
    from ..models import ToProbs

    device = next(model.parameters()).device

    partial_fct = partial(compute_evidential_uncertainties, backend=backend)
    toprobs = ToProbs(activation_fct=activation_fct,
                      loss_type=loss_type, add_constant=add_constant)

    if backend not in ['numpy', 'torch']:
        raise ValueError("Expected backend to be either `numpy` or"
                         f" `torch`. Given {backend}")

    uncertainties = [[] for _ in range(4)]
    with torch.no_grad():
        model.eval()
        for (data, _) in tqdm(loader, desc='evidential uncertainties',
                              leave=False):
            alphas = toprobs(model(data.to(device)).to(torch.double),
                             return_probs=False)
            values = partial_fct(alphas=alphas)

            for (uncer, value) in zip(uncertainties, values):
                uncer.append(value)

    if backend == 'numpy':
        return [np.concatenate(metric, axis=0) for metric in uncertainties]
    else:
        return [torch.cat(metric, dim=0) for metric in uncertainties]
