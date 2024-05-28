import torch
from torch import nn
from tqdm.auto import tqdm

# logger object
import logging
logger_file = logging.getLogger('main_file')


def train_model(model, loader, loss_fct, optimizer,
                nb_samples=None, grad_norm=None, device=None):
    if nb_samples is None:
        nb_samples = len(loader.dataset)

    if device is None:
        device = next(model.parameters()).device

    acc_model, loss_model = 0, 0

    model.train()
    for (data, targets) in tqdm(loader, desc='dataloader train',
                                total=len(loader), leave=False):
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        acc_model += (logits.argmax(dim=1) == targets).sum().item()

        # choose class per batch
        loss = loss_fct(logits, targets)

        loss_model += loss.item() * logits.shape[0]

        optimizer.zero_grad()

        loss.backward()
        # clip gradient
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

    return loss_model/nb_samples, acc_model/nb_samples


def test_model(model, loader, loss_fct=None, device=None, nb_samples=None):
    if nb_samples is None:
        nb_samples = len(loader.dataset)

    if device is None:
        device = next(model.parameters()).device

    if loss_fct is None:
        loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct_pred = 0

        for (data, targets) in tqdm(loader, desc='dataloader eval',
                                    total=len(loader), leave=False):
            data, targets = data.to(device), targets.to(device)

            logits = model(data)

            correct_pred += (logits.argmax(dim=1) == targets).sum().item()
            total_loss += loss_fct(logits, targets).item() * logits.shape[0]

        return total_loss/nb_samples, correct_pred/nb_samples
