import torch
from torch import nn
from tqdm.auto import tqdm

# logger object
import logging
logger_file = logging.getLogger('main_file')


def brier_score(targets, probabilities):
    n_classes = probabilities.shape[-1]

    targets_2d = nn.functional.one_hot(targets, num_classes=n_classes)
    return torch.sum((targets_2d - probabilities)**2, dim=1).mean()


def expected_calibration_error(targets, probabilities, num_bins=15):
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(targets).to(torch.float)

    b = torch.linspace(0, 1.0, num_bins)
    bins = torch.bucketize(confidences, b, right=True)

    ece = torch.tensor(0.)
    for b in range(num_bins):
        mask = bins == b
        if torch.any(mask):
            ece += torch.abs(
                torch.sum(accuracies[mask] - confidences[mask]))

    return ece / targets.shape[0]


def static_calibration_error(targets, probabilities, num_bins=15):
    classes = probabilities.shape[-1]

    sce = torch.tensor(0.)
    for cur_class in range(classes):
        accuracies = (cur_class == targets).to(torch.float)
        confidences = probabilities[..., cur_class].contiguous()

        b = torch.linspace(0, 1.0, num_bins)
        bins = torch.bucketize(confidences, b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if torch.any(mask):
                sce += torch.abs(
                    torch.sum(accuracies[mask] - confidences[mask]))

    return sce / (targets.shape[0] * classes)


def compute_calibration_metrics(model, dataloader, n_bins=15, device=None,
                                temperature=1, return_probs=False):
    '''
    adapted from:
    https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
    https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
    '''
    if device is None:
        device = next(model.parameters()).device

    all_targets = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(dataloader, leave=False,
                                  desc='calibration metrics'):
            data, targets = data.to(device), targets.to('cpu')

            all_targets.append(targets)
            all_probs.append(nn.functional.softmax(
                model(data).to('cpu') / temperature, dim=1))

        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        # ECE computation
        ece = expected_calibration_error(all_targets, all_probs, n_bins)

        # SCE computation
        sce = static_calibration_error(all_targets, all_probs, n_bins)

    if return_probs:
        return ece, sce, all_probs
    else:
        return ece, sce
