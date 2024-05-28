from collections.abc import Iterable

import re
import torch
from torch import nn
import numpy as np
import argparse
from itertools import chain
from pathlib import Path
from functools import partial
from joblib import delayed
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             auc, average_precision_score)
import pandas as pd

from ..models import ToProbs
from ..utils.config import (ConfigYaml, DictToDotNotation,
                            dot_notation_to_dict, get_item_recursively)
from ..utils.utils import equal_dicts, ProgressParallel
from ..data.datasets import train_val_test_datasets, get_dataloaders
from ..metrics.uncertainties import (SoftplusNormalization,
                                     compute_uncertainties,
                                     compute_evidential_uncertainties)
from ..metrics.metrics import (expected_calibration_error,
                               static_calibration_error, brier_score)

reg_float_int = r"[-+]?(?:\d*\.\d+|\d+|\d*\.\d+\-\d+)"


datasets = [['mnist', 'fashionmnist'],
            ['cifar10', 'svhn']]

id_ood_map = dict()
for (x, y) in datasets:
    id_ood_map[x] = y
    id_ood_map[y] = x

to_probs = dict(softmax=nn.Softmax(dim=2),
                softplus=SoftplusNormalization(dim=2))

UNCER_NAMES = ('entropy', 'conditional_entropy', 'mutual_information',
               'total_uncertainty', 'expected_data_entropy',
               'distributional_uncertainty', 'differential_entropy')

loss_fct = nn.CrossEntropyLoss(reduction='sum')
quantiles_per = torch.tensor([0.25, 0.5, 0.75], dtype=torch.double)


def product_dict(id_res: str, d1: dict, id1: str, d2: dict, id2: str) -> dict:
    """
    d1 and d2 have the same structure but different IDs (in the middle):
        id1 vs id2
    """
    f_keys = map(lambda x: x.replace(id1, "{}"), list(d1.keys()))
    return {f_key.format(id_res): d1[f_key.format(id1)] * d2[f_key.format(id2)]
            for f_key in f_keys}


def compute_probs(logits, activation_fct, loss_section=None):
    if loss_section is None or loss_section.get('loss_type') is None:
        loss_section = dict(activation_fct=activation_fct)

    if 'edl' not in loss_section.get('loss_type', '').lower():
        loss_section['activation_fct'] = activation_fct
    # set add_constant to 1.: compatibility with previous trainings
    else:
        loss_section['add_constant'] = loss_section.get('add_constant', 1.)

    toprobs = ToProbs(**loss_section)

    preds = toprobs(logits).to(torch.double)
    # for evidential uncertainties, it makes more sense if logits.shape[1] == 1
    alphas = toprobs(logits, return_probs=False).mean(dim=1).to(torch.double)

    return (preds, alphas)


def uncertainties_fct(logits, activation_fct, dtype='torch',
                      loss_section=None):
    predictions, alphas = compute_probs(logits, activation_fct, loss_section)

    uncertainties = compute_uncertainties(predictions, backend='torch')[2:]

    uncertainties.extend(compute_evidential_uncertainties(
        alphas, backend='torch'))

    if dtype.lower() == 'numpy':
        return [metric.cpu().detach().numpy() for metric in uncertainties]
    else:
        return uncertainties


def get_test_target(dataset, batch_size):
    loader = get_dataloaders(
        batch_size, *train_val_test_datasets(
            dataset=dataset)[:3])[-1]
    return torch.cat([y for _, y in loader])


def load_all_targets(batch_size):
    targets = dict()

    for dataset in id_ood_map:
        loader = get_dataloaders(
            batch_size, *train_val_test_datasets(
                dataset=dataset)[:3])[-1]
        targets[dataset] = torch.cat([y for _, y in loader])

    return targets


def default_info_from_config(config):
    if isinstance(config, DictToDotNotation):
        config = dot_notation_to_dict(config)

    if not isinstance(config, dict):
        raise ValueError(f'expected dict object; found {type(config)}')

    # getter function
    getter = partial(get_item_recursively, config)

    info = dict()
    # data section
    info.update({f'data_section_{k}': v for k, v in getter('data').items()})
    # random seed
    info['random_seed'] = getter('random_seed')
    # loss
    info['loss'] = getter(('loss', 'reg_type'))
    # training epochs
    info['training_epochs'] = getter(('training', 'training_epochs'))
    # model section
    info.update({f'model_section_{k}': tuple(v) if isinstance(v, list) else v
                 for k, v in getter('model').items()
                 if not isinstance(v, dict)})

    return info


def boxplot_stats(uncertainties, is_id_set, mask_good_logits, mask_good_probs):
    stats = dict()
    for (name, value) in zip(UNCER_NAMES, uncertainties):
        stats[f'mean_{name}'] = value.mean().item()
        stats[f'std_{name}'] = value.std().item()

        # https://stackoverflow.com/a/58097941/17773150
        # compute quantiles
        suffixes = ['all', 'good_logits', 'misclass_logits',
                    'good_probs', 'misclass_probs']
        if is_id_set:
            pred_tensors = [value, value[mask_good_logits],
                            value[~mask_good_logits],
                            value[mask_good_probs],
                            value[~mask_good_probs]]
        else:
            pred_tensors = [value] + 4*[None]

        for (suffix, pred_tensor) in zip(suffixes, pred_tensors):
            _suf = f'{name}_{suffix}'
            if pred_tensor is None:
                sub_stats = {f'med_{_suf}': np.nan, f'q1_{_suf}': np.nan,
                             f'q3_{_suf}': np.nan, f'whislo_{_suf}': np.nan,
                             f'whishi_{_suf}': np.nan, f'iqr_{_suf}': np.nan}
            else:
                quantiles = [v.item()
                             for v in pred_tensor.quantile(quantiles_per)]
                iqr = quantiles[2] - quantiles[0]
                sub_stats = {f'med_{_suf}': quantiles[1],
                             f'q1_{_suf}': quantiles[0],
                             f'q3_{_suf}': quantiles[2],
                             f'whislo_{_suf}': quantiles[0] - 1.5*iqr,
                             f'whishi_{_suf}': quantiles[2] + 1.5*iqr,
                             f'iqr_{_suf}': quantiles[2] - quantiles[0]}
            stats.update(sub_stats)
    return stats


def load_logits(logits_files):
    if logits_files is None or not logits_files:
        return None

    if isinstance(logits_files, Iterable):
        logits = []
        for logits_file in logits_files:
            try:
                x = torch.from_numpy(np.load(logits_file))
            except Exception as err:
                raise ValueError(err, logits_file)
            if len(x.shape) == 2:
                logits.append(x.unsqueeze(dim=1))
            else:
                logits.append(x)
        logits = torch.cat(logits, dim=1)
    else:
        try:
            logits = torch.from_numpy(np.load(logits_files))
        except Exception as err:
            raise ValueError(err, logits_files)
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(dim=1)
    return logits


def check_if_training_stopped(path_model_folder):
    """
    function to check if the training of the model has stopped
    by reading the logs file
    """
    with open(path_model_folder / 'logs.log') as f:
        for line in f:
            if 'NaN' in line:
                return True
    return False


def get_al_set_size(path_model_folder):
    with open(path_model_folder / 'logs.log', 'r') as f:
        for line in f:
            if 'len_labeled_dataset' in line:
                line = line.split(' ')
                return int(re.findall(reg_float_int, line[-2])[0])
    return None


@torch.no_grad()
def one_compute_auc(suffix, neg_value, pos_value):
    y_true = ([0 for _ in range(len(neg_value))]
              + [1 for _ in range(len(pos_value))])
    y_score = np.concatenate((neg_value, pos_value))

    try:
        auc_v = roc_auc_score(y_true, y_score)
        pr, recall, _ = precision_recall_curve(y_true, y_score)
        aupr = auc(recall, pr)
        ap = average_precision_score(y_true, y_score)
        sub_row = {f'auc_{suffix}': auc_v, f'aupr_{suffix}': aupr,
                   f'ap_{suffix}': ap}
    except Exception:
        sub_row = {f'auc_{suffix}': np.nan, f'aupr_{suffix}': np.nan,
                   f'ap_{suffix}': np.nan}
    return sub_row


@torch.no_grad()
def compute_auc(prefix, neg_uncertainties, pos_uncertainties,
                loss_section, same_length=True):
    """
    negative logits: those for which score should be low
    positive logits: those for which score should be high
    """
    n_samples = (min(len(neg_uncertainties[0]), len(pos_uncertainties[0]))
                 if same_length else
                 max(len(neg_uncertainties[0]), len(pos_uncertainties[0])))

    auc_metrics = dict()
    for (name, neg_value, pos_value) in zip(
            UNCER_NAMES, neg_uncertainties, pos_uncertainties):
        _suf = f'{prefix}_{name}'

        sub_row = one_compute_auc(_suf, neg_value[:n_samples],
                                  pos_value[:n_samples])
        auc_metrics.update(sub_row)

    return auc_metrics


def model_default_performance(logits, is_id_set, targets,
                              activation_fct, loss_section):
    mean_logits = logits.mean(dim=1)
    predictions, _ = compute_probs(logits, activation_fct, loss_section)
    uncertainties = uncertainties_fct(logits, activation_fct,
                                      dtype='torch', loss_section=loss_section)

    if is_id_set:
        nll_loss = loss_fct(mean_logits, targets).item()
        # accuracy based on the mean of the logits
        good_pred_logits = (mean_logits.argmax(dim=1) == targets
                            ).nonzero().flatten().tolist()
        mask_good_logits = np.isin(range(len(logits)),
                                   good_pred_logits)
        accuracy_logits = len(good_pred_logits)
        # accuracy based on the mean of the probabilities
        good_pred_probs = (predictions.mean(dim=1).argmax(dim=1) == targets
                           ).nonzero().flatten().tolist()
        mask_good_probs = np.isin(range(len(logits)), good_pred_probs)
        accuracy_probs = len(good_pred_probs)
        # calibration metrics
        mean_predictions = predictions.mean(dim=1)
        ece = expected_calibration_error(targets, mean_predictions).item()
        sce = static_calibration_error(targets, mean_predictions).item()
        brier = brier_score(targets, mean_predictions).item()
    else:
        nll_loss, accuracy_logits, accuracy_probs = 0, 0, 0
        mask_good_logits, mask_good_probs = None, None
        ece, sce, brier = None, None, None

    # boxplot stats
    stats = boxplot_stats(uncertainties, is_id_set,
                          mask_good_logits, mask_good_probs)
    # update row dict
    metrics = dict(nll_loss=nll_loss / len(logits),
                   accuracy_probs=accuracy_probs / len(logits),
                   accuracy_logits=accuracy_logits / len(logits),
                   ece=ece, sce=sce, brier=brier, **stats)

    return (metrics, uncertainties, mask_good_probs)


@torch.no_grad()
def compute_model_perf(loss_section, targets, id_logits_files=None,
                       ood_logits_files=None):
    id_logits = load_logits(id_logits_files).to(torch.float)
    ood_logits = load_logits(ood_logits_files).to(torch.float)

    results = []
    for activation_fct in to_probs:
        _params = (targets, activation_fct, loss_section)

        if id_logits is not None:
            id_row = dict(activation=activation_fct, is_id=True)
            id_metrics, id_uncer, m_good_probs = model_default_performance(
                id_logits, True, *_params)
            id_row.update(id_metrics)
            id_good_uncer = [uncer[m_good_probs] for uncer in id_uncer]
            id_mis_uncer = [uncer[~m_good_probs] for uncer in id_uncer]
            auc_mis = compute_auc('mis', id_good_uncer, id_mis_uncer,
                                  loss_section, False)
            id_row.update(auc_mis)
        if ood_logits is not None:
            ood_row = dict(activation=activation_fct, is_id=False)
            ood_metrics, ood_uncer, _ = model_default_performance(
                ood_logits, False, *_params)
            ood_row.update(ood_metrics)

        if id_logits is not None and ood_logits is not None:
            # TODO: set same_length to False
            id_row.update(compute_auc('ood', id_uncer, ood_uncer,
                                      loss_section, True))
            id_row.update(compute_auc('good_vs_ood', id_good_uncer,
                                      ood_uncer, loss_section, False))
            auc_mis_ood = compute_auc('mis_vs_ood', id_mis_uncer,
                                      ood_uncer, loss_section, False)
            id_row.update(auc_mis_ood)
            # AUC(good, mis, OOD) = AUC(good, mis) * AUC(mis, OOD)
            auc_good_mis_ood = product_dict(
                "good_vs_mis_vs_ood", auc_mis, "mis",
                auc_mis_ood, "mis_vs_ood")
            id_row.update(auc_good_mis_ood)

        if id_logits is not None:
            results.append(id_row)
        if ood_logits is not None:
            results.append(ood_row)

    return results


def process_one_folder(path_model_folder, targets=dict(), batch_size=1000,
                       config_extractor=default_info_from_config):
    """
    this function is to be used if:
        * the model saved in the checkpoint is an ensemble
        * the logits are generated using MC-Dropout
    """
    path_model_folder = Path(path_model_folder).resolve().absolute()

    if path_model_folder.is_file():
        path_model_folder = path_model_folder.parent

    config = ConfigYaml(path_model_folder / 'config.yaml').dict_config

    row = dict(date=path_model_folder.parent.stem,
               time=path_model_folder.stem,
               parent=path_model_folder.parents[1].stem,
               al_set_size=get_al_set_size(path_model_folder),
               is_stopped=check_if_training_stopped(path_model_folder))
    row.update(config_extractor(config))

    train_set = row.get('data_section_dataset')
    if train_set is None:
        raise ValueError(f'the function {config_extractor} should return'
                         ' a dict that has the key: data_section_dataset')

    folder_results = []

    # process id and ood samples
    id_logits_files = list(path_model_folder.glob('**/id*samples.npy'))
    ood_logits_files = list(path_model_folder.glob('**/ood*samples.npy'))

    if train_set not in targets:
        targets[train_set] = get_test_target(train_set, batch_size)

    results = compute_model_perf(config['loss'], targets[train_set],
                                 id_logits_files, ood_logits_files)

    if isinstance(results, list):
        folder_results.extend([{**row, **res} for res in results])
    elif isinstance(results, dict):
        folder_results.append({**row, **results})
    else:
        raise ValueError(f'expected a dict - found {type(results)}')

    return folder_results


def process_list_folders(list_folders_path, all_targets,
                         config_extractor=default_info_from_config):
    # TODO: to be tested :)
    """
    this function is to be used if:
        * the ensemble models are trained separately and saved
        in different checkpoints
        * OR, more generally, for any list of models trained on the same set
    """
    # convert to Path
    list_folders_path = [Path(folder).resolve().absolute()
                         for folder in list_folders_path]
    # get parent folder
    list_folders_path = [folder.parent if folder.is_file() else folder
                         for folder in list_folders_path]

    # get config params from the first folder path
    config = ConfigYaml(list_folders_path[0] / 'config.yaml').dict_config
    row = config_extractor(config)

    train_set = row.get('data_section_dataset')
    if train_set is None:
        raise ValueError(f'the function {config_extractor} should return'
                         ' a dict that has the key: [data]_dataset')

    for path_model_folder in list_folders_path[1:]:
        config = ConfigYaml(path_model_folder / 'config.yaml').config
        # except for the random seed, we expect the same config params
        assert equal_dicts(row, config_extractor(config),
                           ignore_keys=('random_seed',))

    id_files = list(chain(*[folder.glob('**/id*samples.npy')
                            for folder in list_folders_path]))
    ood_files = list(chain(*[folder.glob('**/ood*samples.npy')
                             for folder in list_folders_path]))

    results = compute_model_perf(config['loss'], all_targets[train_set],
                                 id_files, ood_files)

    if isinstance(results, list):
        return [{**row, **res} for res in results]
    elif isinstance(results, dict):
        return [{**row, **results}]
    else:
        raise ValueError(f'expected a dict - found {type(results)}')


if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser('eval_aggregate_results')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=10)
    parser.add_argument('--csv-file', type=str, default='df')
    parser.add_argument('folders', nargs='*')
    args = parser.parse_args()

    # list of all checkpoints
    ckpt_paths = list(
        chain(*[Path(folder).resolve().absolute().glob('**/checkpoint.pth')
                for folder in args.folders]))

    main_fct = partial(process_one_folder, batch_size=args.batch_size)

    with torch.no_grad():
        # Parallel object
        if args.jobs > 1:
            par_obj = ProgressParallel(use_tqdm=True, total=len(ckpt_paths),
                                       n_jobs=args.jobs)
            results = par_obj(delayed(main_fct)(ckpt_path)
                              for ckpt_path in ckpt_paths)
        else:
            # load all targets
            all_targets = load_all_targets(args.batch_size)

            results = [main_fct(ckpt_path, targets=all_targets)
                       for ckpt_path in ckpt_paths]

        # flatten list
        results = [row for res in results for row in res]

        # save to csv file
        df = pd.DataFrame(results)
        project_path = Path(__file__).resolve().parents[2].absolute()
        df.to_csv(project_path / f'{args.csv_file}.csv',
                  index=False, header=True)
