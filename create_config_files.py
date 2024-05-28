from tqdm.auto import tqdm
from pathlib import Path
from functools import reduce
from itertools import product

from src.utils.config import ConfigYaml

USE_WD = True
DATA_WD = False

path_exp = Path(__file__).resolve().absolute().parent
base_config = path_exp / 'base_config.yaml'
# replace this with path where logs will be saved
logs_folder = "/tmp/logs/"


if not logs_folder.endswith('/'):
    # the path should end with "/"
    logs_folder += '/'

if USE_WD:
    logs_folder += "low_wd"
else:
    logs_folder += "no_wd"

data_mapping = dict(
    mnist=dict(optimizer=dict(opt_class='SGD',
                              params=dict(lr=0.01, momentum=0.95)),
               training_epochs=500),
    svhn=dict(optimizer=dict(opt_class='SGD',
                             params=dict(lr=0.02, momentum=0.95)),
              training_epochs=600),
    cifar10=dict(optimizer=dict(opt_class='SGD',
                                params=dict(lr=0.04, momentum=0.9)),
                 training_epochs=700),
)


ratios = [0.005, 0.008, 0.013, 0.02, 0.033, 0.052,
          0.084, 0.134, 0.215, 0.344, 0.55, 0.88, 1]
weight_decays = [round(3e-5 / r**(2/3), 5) for r in ratios]

hidden_layers = [
    [2*i, i] for i in map(lambda x: 2**x, range(6, 11))
]

loss_mapping = dict(
    mcdropout_CE=dict(reg_type=None, ce_params=dict(),
                      reg_params=dict()),
    mcdropout_LS=dict(reg_type='LS', ce_params=dict(),
                      reg_params=dict(reg_coef=0.1)),
    mcdropout_CP=dict(reg_type='CP', ce_params=dict(),
                      reg_params=dict(reg_coef=0.1)),
    EDL=dict(loss_type='EDL', kl_coef=0.01, epoch_update=True,
             activation_fct='softplus'),
    ensemble_CE=dict(reg_type=None, ce_params=dict(),
                     reg_params=dict()),
    ensemble_CL=dict(reg_type='CL', ce_params=dict(),
                     reg_params=dict(reg_coef=0.05)),
    ensemble_LS=dict(reg_type='LS', ce_params=dict(),
                     reg_params=dict(reg_coef=0.1)),
)


if DATA_WD:
    iters = [data_mapping, loss_mapping, hidden_layers,
             list(zip(ratios, weight_decays))]
else:
    iters = [data_mapping, loss_mapping, ratios,
             list(zip(hidden_layers, weight_decays))]

iterator = product(*iters)
total = reduce(lambda x, y: x*y, map(len, iters))

if __name__ == "__main__":
    for (dataset, loss, *x) in tqdm(iterator, total=total):
        if DATA_WD:
            h_l, (ratio, wd) = x
        else:
            ratio, (h_l, wd) = x

        dest_folder = f"mlp/{dataset}/{loss}"
        str_out = '_'.join(map(lambda x: f"{x:04d}", h_l))
        date_suffix = f"ratio_{int(1000*ratio):04d}_h_l_{str_out}"
        config = ConfigYaml(base_config)
        config_dict = config.dict_config.copy()

        # logger
        config_dict['loggers']['date_suffix'] = date_suffix
        config_dict['loggers']['path_logs'] = f"{logs_folder}/{dest_folder}"

        # data
        config_dict['data']['dataset'] = dataset
        config_dict['data']['ratio_subset_train'] = ratio
        _batch_size = 100 + 156 * int(ratio >= 0.01)
        config_dict['data']['batch_size'] = [
            _batch_size, *config_dict['data']['batch_size'][1:]]
        if dataset == "cifar10":
            config_dict['data']['embeddings'] = "ResNet34"
        else:
            config_dict['data']['embeddings'] = "identity"

        # optimizer
        config_dict['optimizer'] = data_mapping[dataset]['optimizer']

        if USE_WD:
            # adjust wd for MNIST
            wd /= 10**(int(dataset == 'mnist'))
            config_dict['optimizer']['params']['weight_decay'] = wd

        # hidden layers
        config_dict['model']['hidden_layers'] = h_l

        # num models
        config_dict['model']['num_models'] = 10**(int('ensemble' in loss))

        # epochs
        epochs = data_mapping[dataset]['training_epochs']
        config_dict['training']['training_epochs'] = epochs
        tolerance = 100 + 100*int(dataset == "cifar10")
        config_dict['training']['early_stopping']['tolerance'] = tolerance

        # loss
        config_dict['loss'] = loss_mapping[loss]

        # write the config file
        config.update(config_dict)
        (path_exp / dest_folder).mkdir(parents=True, exist_ok=True)
        _fname = f"exp_{date_suffix}.yaml"
        config.write_config(path_exp / dest_folder / _fname)
