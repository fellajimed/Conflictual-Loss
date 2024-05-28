import torch
from torch import nn
import numpy as np
import argparse
from random import shuffle
from itertools import chain
from pathlib import Path
from joblib import delayed
from functools import partial
from copy import deepcopy
from tqdm.auto import tqdm
from collections.abc import Iterable

from ..utils.config import ConfigYaml, dot_notation_to_dict
from ..utils.utils import ProgressParallel, get_device
from ..models.ensemble import EnsembleNet
from ..models.utils import model_loader_from_ckpt
from ..data.datasets import train_val_test_datasets, get_dataloaders
from ..metrics.uncertainties import enable_dropout


datasets = [['mnist', 'fashionmnist'],
            ['cifar10', 'svhn']]

id_ood_map = dict()
for (x, y) in datasets:
    id_ood_map[x] = y
    id_ood_map[y] = x


def get_test_loader(data_section, batch_size):
    test_dataset = train_val_test_datasets(**data_section)[2]
    return get_dataloaders(batch_size, test_dataset=test_dataset)[-1]


def model_generator(model, n, mc_dropout=False):
    if n is not None and n > 1 and mc_dropout:
        for _ in range(n):
            yield enable_dropout(model)
    else:
        yield model


def compute_logits(model, loader, device):
    if ((isinstance(model, nn.DataParallel)
         and isinstance(model.module, EnsembleNet))
            or isinstance(model, EnsembleNet)):
        return torch.cat(
            [model(data.to(device),
                   return_mean=False).to(torch.double).cpu().detach()
             for (data, _) in loader], dim=0)
    else:
        return torch.cat(
            [model(data.to(device)).to(torch.double).cpu().detach()
             for (data, _) in loader], dim=0).unsqueeze(dim=1)


def save_logits(list_logits, list_paths):
    """
    list_logits and list_paths are either:
    * a list of (lists of) logits and a list of paths
    OR
    * one (list of) logits and a single path
    """
    if isinstance(list_paths, Iterable):
        assert len(list_logits) == len(list_paths)
        for (logits, path) in zip(list_logits, list_paths):
            save_logits(logits, path)

    elif list_logits != []:
        list_logits = torch.cat(list_logits, dim=1).numpy()
        np.save(list_paths, list_logits)
        del list_logits


def process_one_ckpt(ckpt_path, n, is_id, is_ood, is_mc_dropout,
                     batch_size=1000, use_latest=False,
                     force=False, device=torch.device('cpu')):
    dataloaders = dict()
    folder = ckpt_path.parent

    # create logits folder
    if is_id or is_ood:
        _suffix = (f'_mc_dropout_{n}'
                   if n is not None and is_mc_dropout else
                   f'_sampled_{n}' if n is not None else '')
        logits_folder = folder / f'logits{_suffix}'
        logits_folder.mkdir(parents=True, exist_ok=True)

    config = ConfigYaml(folder / 'config.yaml').config
    train_set = config.data.dataset
    ood_set = id_ood_map[train_set]

    id_data_section = dot_notation_to_dict(config.data)

    ood_data_section = deepcopy(id_data_section)
    ood_data_section['dataset'] = ood_set
    ood_data_section['id_dataset'] = train_set

    all_logits = [[], []]
    list_paths = [logits_folder / f'{val}_{train_set}_samples'
                  for val in ('id', 'ood')]
    is_compute = [val and (not f_path.with_suffix('.npy').is_file() or force)
                  for (val, f_path) in zip((is_id, is_ood), list_paths)]

    if not any(is_compute):
        return

    # load the model
    loaded_model = model_loader_from_ckpt(ckpt_file=ckpt_path,
                                          device=device)[int(use_latest)]
    loaded_model.to(device)
    model_gen = model_generator(loaded_model, n, is_mc_dropout)

    for model in tqdm(model_gen, total=n, leave=False, position=1):
        iter = zip((id_data_section, ood_data_section), is_compute, all_logits)
        for (d_section, _compute_logits, logits) in iter:
            set_name = d_section['dataset']
            if _compute_logits:
                if set_name not in dataloaders:
                    dataloaders[set_name] = get_test_loader(d_section,
                                                            batch_size)
                logits.append(compute_logits(model, dataloaders[set_name],
                                             device))

    save_logits(all_logits, list_paths)

    del loaded_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser('eval_test_logits')
    parser.add_argument('--id', action='store_true')
    parser.add_argument('--ood', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--use-latest', action='store_true')
    parser.add_argument('--mc-dropout', action='store_true')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('folders', nargs='*')
    args = parser.parse_args()

    ckpt_paths = list(
        chain(*[Path(folder).resolve().absolute().glob('**/checkpoint.pth')
                for folder in args.folders]))

    if (args.id or args.ood) and ckpt_paths:
        # shuffle list
        shuffle(ckpt_paths)

        params = dict(n=args.n, is_mc_dropout=args.mc_dropout, is_id=args.id,
                      is_ood=args.ood, batch_size=args.batch_size,
                      force=args.force, use_latest=args.use_latest)
        main_fct = partial(process_one_ckpt, **params)

        if args.jobs > 1:
            # ProgressParallel object
            par_obj = ProgressParallel(use_tqdm=True, total=len(ckpt_paths),
                                       n_jobs=args.jobs)
            par_obj(delayed(main_fct)(ckpt_path) for ckpt_path in ckpt_paths)
        else:
            device = get_device()
            for ckpt_path in tqdm(ckpt_paths):
                main_fct(ckpt_path, device=device)
