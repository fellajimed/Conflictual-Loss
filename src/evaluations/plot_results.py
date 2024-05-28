import re
import argparse
import builtins
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from seaborn.utils import relative_luminance
from joblib import delayed
from ..utils.utils import ProgressParallel


AUC_METRICS = ('mis', 'ood')
uncer_names = ('entropy', 'conditional_entropy', 'mutual_information',
               'total_uncertainty', 'expected_data_entropy',
               'distributional_uncertainty', 'differential_entropy')
id_metrics = ('nll_loss', 'exp_nll_loss', 'brier',
              'accuracy_probs', 'ece', 'sce')


def folder_name(metric_name, is_diff):
    name = None
    if metric_name in ('ece', 'sce'):
        name = "calibration"
    elif metric_name in map(lambda x: f'mean_{x}', uncer_names):
        if is_diff:
            name = "diff_id_ood"
        else:
            name = "uncertainties"
    else:
        for (measure, det_type) in product(('auc', 'aupr', 'ap'), AUC_METRICS):
            if metric_name.startswith(f"{measure}_{det_type}"):
                name = f"{measure}_{det_type}"
                break
    return name


def compute_vmin_vmax(metric_name, df_all, dataset_name,
                      is_id=True, activation_fct='softmax'):
    _df = df_all.query('data_section_dataset == @dataset_name'
                       ' and activation == @activation_fct'
                       ' and is_id == @is_id')
    values = _df[metric_name]
    return (values.min(), values.max())


def get_data(metric_name, df, x_axis, y_axis,
             is_id=True, activation_fct='softmax'):
    # is_id in [True, False]
    # activation_fct in ['softmax', 'softplus']

    df_plots = df[(df.activation == activation_fct)
                  & (df.is_id == is_id)]

    if isinstance(df_plots[y_axis], pd.Series):
        df_plots = df_plots.sort_values([y_axis, 'tmp_x_axis'],
                                        ascending=[False, True])
        y_label = df_plots[y_axis].unique()
    else:
        df_plots = df_plots.sort_values([(y_axis, 'mean'), 'tmp_x_axis'],
                                        ascending=[False, True])
        y_label = df_plots[y_axis]['mean'].unique()

    x_label = (df_plots[x_axis].unique()
               if isinstance(df_plots[x_axis], pd.Series)
               else df_plots[x_axis]['mean'].unique())

    data = (df_plots[metric_name].values
            if isinstance(df_plots[metric_name], pd.Series)
            else df_plots[metric_name]['mean'].values
            ).reshape(len(y_label), len(x_label)).squeeze()
    std = (0. if isinstance(df_plots[metric_name], pd.Series)
           else df_plots[metric_name]['std'].values.reshape(
            len(y_label), len(x_label)).squeeze())

    return (data, std, x_label, y_label)


def get_data_diff(metric_name, df, x_axis, y_axis,
                  activation_fct='softmax', **kwargs):
    data_id, _, x_label_id, y_label_id = get_data(
            metric_name, df, x_axis, y_axis, True, activation_fct)

    data_ood, _, x_label_ood, y_label_ood = get_data(
            metric_name, df, x_axis, y_axis, False, activation_fct)

    assert ((x_label_id == x_label_ood).all()
            and (y_label_id == y_label_ood).all())

    return (data_id - data_ood, x_label_id, y_label_id)


def center_values(vmin, vmax, vcenter, eps=0.05):
    if vcenter is None:
        return vmin, vmax

    threshold = vcenter - eps
    vmin = vmin*int(vmin < threshold) + threshold*int(vmin >= threshold)
    threshold = vcenter + eps
    vmax = vmax*int(vmax > threshold) + threshold*int(vmax <= threshold)
    return vmin, vmax


def ax_text(ax, im, data, fmt):
    for (j, i), val in np.ndenumerate(data):
        # inspired by: seaborn.matrix ...
        lum = relative_luminance(im.cmap(im.norm(val)))
        text_color = ".15" if lum > .408 else "w"
        ax.text(i, j, f'{val:{fmt}}', color=text_color,
                fontsize='small', ha='center', va='center')


def plot_1d(data, vmin, vmax, x_label):
    if vmin is None or vmax is None:
        vmin = np.amin(data)
        vmax = np.amax(data)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    ax.plot(x_label, data, '-o')
    ax.set_ylim([vmin - abs(vmin)*0.05, vmax + abs(vmax)*0.05])
    ax.set_xscale("linear")
    return fig, ax


def plot_2d(data, vmin, vmax, vcenter, fmt):
    if vmin is None or vmax is None:
        vmin = np.amin(data)
        vmax = np.amax(data)

    if vcenter is None:
        im_kwargs = dict(vmin=vmin, vmax=vmax, cmap='inferno')
    else:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        im_kwargs = dict(norm=norm, cmap='seismic_r')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8))
    im = ax.imshow(data, interpolation="quadric", aspect='auto', **im_kwargs)
    ax_text(ax, im, data, fmt)
    plt.colorbar(im, ax=ax)

    return fig, ax


def plot(metric_name, df_all, dataset_name, reg_type, figures_path,
         is_id, x_axis, y_axis, is_normalized=False,
         activation_fct='softmax', is_diff=False, suf_name=''):
    df = df_all.query('data_section_dataset == @dataset_name'
                      ' and date_suffix == @reg_type')

    vcenter = 0.5 if 'auc' in metric_name else 0. if is_diff else None

    if is_diff:
        data, x_label, y_label = get_data_diff(
            metric_name, df, x_axis, y_axis, activation_fct)
    else:
        data, _, x_label, y_label = get_data(
            metric_name, df, x_axis, y_axis, is_id, activation_fct)

    if np.isnan(data).all():
        return

    if is_normalized:
        vmin, vmax = compute_vmin_vmax(
            metric_name, df_all, dataset_name, is_id, activation_fct)
        normalization = "normalized"
    else:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        normalization = "default"

    fmt = ('.2%' if any(m in metric_name
                        for m in ['accuracy', 'auc', 'ap', 'aupr'])
           else '.2e')

    vmin, vmax = center_values(vmin, vmax, vcenter)

    fig_path = figures_path / dataset_name / reg_type / normalization
    folder = folder_name(metric_name, is_diff)
    if folder is not None:
        fig_path = fig_path / folder
    fig_path.mkdir(parents=True, exist_ok=True)

    if len(data.shape) == 1:
        fig, ax = plot_1d(data, vmin, vmax, x_label)
    else:
        fig, ax = plot_2d(data, vmin, vmax, vcenter, fmt)
        ax.set_yticks(range(len(y_label)))
        ax.set_yticklabels(y_label)
        ax.set_xticks(range(len(x_label)))
        ax.set_xticklabels(x_label)

    ax.set_title('')
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(fig_path / f"{metric_name}{suf_name}.pdf")
    plt.close()


def main_fct(metric, set_name, norm, reg_type, kwargs):
    kwargs.update(dict(metric_name=metric, dataset_name=set_name,
                       is_normalized=norm, reg_type=reg_type, is_diff=False))
    if (metric in id_metrics
            or any(v in metric for v in ('auc_', 'aupr_', 'ap_'))):
        kwargs.update(dict(is_id=True))
        plot(**kwargs)
    else:
        for is_id in (True, False):
            kwargs.update(dict(is_id=is_id,
                               suf_name='_id' if is_id else '_ood'))
            plot(**kwargs)

    if 'mean_' in metric:
        kwargs.update(dict(is_diff=True))
        plot(**kwargs)


def total_len_train_set(dataset):
    # after a 20% split
    if dataset.lower() == 'cifar10':
        return 40000
    elif dataset.lower() in ['mnist', 'fashionmnist']:
        return 48000
    elif dataset.lower() == 'svhn':
        return 58606
    else:
        raise ValueError('check train set')


reg_float_int = r"[-+]?(?:\d*\.\d+|\d+|\d*\.\d+\-\d+)"


def to_right_tuple_type(df, col, element_type=int):
    try:
        df.loc[~df[col].isna(), col] = df[col][~df[col].isna()].apply(
            lambda x: tuple(map(element_type, re.findall(reg_float_int, x))))
    except Exception as e:
        print('Exception:', col, e)


def process_results_dataframe(csv_file, x_axis, x_type, y_axis,
                              y_type, is_parent=False):
    df_all = pd.read_csv(csv_file)

    # new columns: len_train_set, exp_nll_loss
    df_all['len_train_set'] = df_all.apply(
        lambda row: int(row.data_section_ratio_subset_train
                        * total_len_train_set(row.data_section_dataset)),
        axis=1)
    if 'exp_nll_loss' not in df_all:
        df_all['exp_nll_loss'] = np.exp(-df_all['nll_loss'])

    # objects to tuples
    to_right_tuple_type(df_all, 'model_section_input_shape', int)
    to_right_tuple_type(df_all, 'model_section_hidden_layers', int)
    to_right_tuple_type(df_all, 'model_section_planes', int)
    to_right_tuple_type(df_all, 'model_section_dropout_rates', float)

    to_right_tuple_type(df_all, x_axis, getattr(builtins, x_type))
    to_right_tuple_type(df_all, y_axis, getattr(builtins, y_type))

    # drop nan columns
    df_all.dropna(how='all', inplace=True, axis=1)

    # unique experiments are based on date suffix
    if is_parent:
        df_all['date_suffix'] = df_all['parent']
    elif 'date' in df_all.columns:
        df_all['date_suffix'] = df_all['date'].str[9:]
    elif 'loggers_section_date_suffix' in df_all.columns:
        df_all['date_suffix'] = df_all['loggers_section_date_suffix']

    # tmp x axis
    if isinstance(df_all[x_axis][0], str):
        df_all['tmp_x_axis'] = df_all[x_axis].str.extract(
            r'(\d+)', expand=False).astype(int)
    else:
        df_all['tmp_x_axis'] = df_all[x_axis]

    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser('plot_results')
    parser.add_argument('-x', '--x-axis', type=str,
                        default='model_section_hidden_layers')
    parser.add_argument('--x-type', type=str, default='int')
    parser.add_argument('-y', '--y-axis', type=str,
                        default='len_train_set')
    parser.add_argument('--y-type', type=str, default='float')
    parser.add_argument('-m', '--metrics', nargs="+", default=['all'])
    parser.add_argument('-d', '--datasets', nargs="+", default=['all'])
    parser.add_argument('-j', '--jobs', type=int, default=10)
    parser.add_argument('-f', '--csv-files', nargs="+", type=str)
    parser.add_argument('--parent', action='store_true')
    parser.add_argument('-p', '--figures-path', type=str, default=None)
    args = parser.parse_args()

    df_all = pd.concat([
        process_results_dataframe(csv_file, args.x_axis, args.x_type,
                                  args.y_axis, args.y_type, args.parent)
        for csv_file in args.csv_files], ignore_index=True)

    if len(args.datasets) == 1 and args.datasets[0].lower() == 'all':
        args.datasets = df_all.data_section_dataset.unique()

    if len(args.metrics) == 1 and args.metrics[0].lower() == 'all':
        args.metrics = list(id_metrics)
        for uncer in uncer_names:
            col_name = f"mean_{uncer}"
            if col_name in df_all.columns:
                args.metrics.append(col_name)

        for (measure, det_type, uncer) in product(('auc', 'aupr', 'ap'),
                                                  AUC_METRICS, uncer_names):
            col_name = f"{measure}_{det_type}_{uncer}"
            if col_name in df_all.columns:
                args.metrics.append(col_name)

    # figures path
    if args.figures_path is None:
        args.figures_path = Path(__file__).parent.resolve().absolute()
    else:
        args.figures_path = Path(args.figures_path).resolve().absolute()
    args.figures_path.mkdir(parents=True, exist_ok=True)

    unique_exp = df_all['date_suffix'].unique()

    kwargs = dict(df_all=df_all, figures_path=args.figures_path,
                  x_axis=args.x_axis, y_axis=args.y_axis,
                  activation_fct='softmax')

    n_iter = 2 * len(args.metrics) * len(args.datasets) * len(unique_exp)

    iterator = product(args.metrics, args.datasets,
                       (True, False), unique_exp)

    if args.jobs > 1:
        # ProgressParallel object
        par_obj = ProgressParallel(use_tqdm=True, total=n_iter,
                                   n_jobs=args.jobs)
        par_obj(delayed(main_fct)(metric, set_name, norm, reg_type, kwargs)
                for (metric, set_name, norm, reg_type) in iterator)
    else:
        for (metric, set_name, norm, reg_type) in tqdm(iterator, total=n_iter):
            main_fct(metric, set_name, norm, reg_type, kwargs)
