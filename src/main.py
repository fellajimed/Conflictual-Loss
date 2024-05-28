'''
main file
'''
# imports
import argparse
import copy
import shutil
import time
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from pathlib import Path
from functools import partial
from codecarbon import EmissionsTracker

from . import init_logger
from . import models
from .data.datasets import train_val_test_datasets, get_dataloaders
from .utils.utils import get_device, setup_seed
from .utils.config import ConfigYaml, dot_notation_to_dict, DictToDotNotation
from .loss import Loss
from .trainers.trainer import ModelTrainer
from .evaluations.compute_logits import id_ood_map


def init_config(config_file):
    # config file
    config_yaml = ConfigYaml(config_file)
    config = config_yaml.config

    # create loggers
    logs_kwargs = (dot_notation_to_dict(config.loggers)
                   if config.loggers is not None else dict())
    logger, _, _, path_logs, final_dest = init_logger(**logs_kwargs)

    # save the config file to the log folder associated to this run
    config_yaml.write_config(path_logs / 'config.yaml')

    logger.info(f"path to config file: {config_yaml.path}\nA copy of the"
                " config file could be found in the logs directory"
                f" {path_logs}/config.yaml\nUsing the"
                f" following configuration:\n{config_yaml.pretty_print()}")

    return config, logger, path_logs, final_dest


def init_data(config, logger, device):
    # get random seed
    random_seed = config.random_seed if config.random_seed is not None else 42

    # the dataset is saved in ./data
    data_section = dot_notation_to_dict(config.data)
    data_section['dataset'] = data_section.get('dataset', 'mnist')
    data_section['device'] = device

    # fix random seed for data
    data_random_seed = config.data.random_seed
    if data_random_seed is not None:
        logger.info("setting the random seed before getting the datasets")
        setup_seed(data_random_seed)
    else:
        setup_seed(random_seed)
    # get datasets
    *sets, input_shape, nb_classes = train_val_test_datasets(**data_section)

    logger.info(f"Number of samples per dataset: training={len(sets[0])}"
                f" - validation={len(sets[1])} - test={len(sets[2])}")

    # apply `random_seed` in the remaining code
    if data_random_seed is not None:
        setup_seed(random_seed)

    return *sets, input_shape, nb_classes


def init_model(config, input_shape, nb_classes, device, logger):
    # model params from the yaml file
    model_params = dot_notation_to_dict(config.model)
    model_class = getattr(models, model_params.get('model_class', 'MLPNet'))
    if input_shape is not None:
        model_params['input_shape'] = input_shape
        model_params['nb_classes'] = nb_classes
    model_params['model_class'] = model_class
    model_params['num_models'] = model_params.get('num_models', 1)

    # model definition
    model = models.EnsembleNet(device=device, **model_params)

    # model summary
    try:
        model_summary = summary(model, (1, *model.input_shape), verbose=0)
        logger.info(f"model summary:\n{str(model_summary)}")
    except Exception:
        logger.warn('could not print summary using torchinfo')
    logger.info(f"model print:\n{model}")

    return model, model_params, model_class


def init_loss(config, model, model_params, len_train_set, logger):
    loss_config = config.loss
    if loss_config is None:
        loss_fct = nn.CrossEntropyLoss()
    else:
        loss_params = dot_notation_to_dict(loss_config).copy()
        loss_params['reg_params'] = loss_params.get('reg_params', dict())
        if (loss_config.loss_type is not None
                and 'elbo' in loss_config.loss_type.lower()):
            len_train_set = loss_params.get('len_train_set', len_train_set)
            loss_params['len_train_set'] = len_train_set
        else:
            len_train_set = loss_params['reg_params'].get('len_train_set',
                                                          len_train_set)
            loss_params['reg_params']['len_train_set'] = len_train_set

        if (loss_params.get('reg_type') in ['CR', 'DCR', 'CRLN']
                and isinstance(model, models.EnsembleNet)):
            nb_classes = model_params['nb_classes']
            # define a list of loss functions
            loss_fct = []
            for i in range(nb_classes):
                _params = copy.deepcopy(loss_params)
                _params['reg_params']['class_index'] = i % nb_classes
                loss_fct.append(Loss(**_params))
        else:
            loss_fct = Loss(**loss_params.copy())
    logger.info(f"loss function print:\n{loss_fct}")

    return loss_fct


def init_optimizer(config, logger):
    optimizer_config = config.optimizer
    # default optmizer: SGD
    opt_params = dict(lr=1e-3)
    opt_class = torch.optim.SGD
    if optimizer_config is not None:
        if optimizer_config.opt_class is not None:
            opt_class = getattr(torch.optim, optimizer_config.opt_class,
                                opt_class)

        # update the optimizer params from the config file
        _params = (dict() if optimizer_config.params is None
                   else dot_notation_to_dict(optimizer_config.params))
        opt_params.update(_params)

    optimizer = partial(opt_class, **opt_params)
    logger.info(f"optimizer print:\n{optimizer}")

    return optimizer


def init_lr_scheduler(config, logger):
    lr_scheduler_config = config.lr_scheduler
    # by default, no learning rate scheduler is applied
    if ((lr_scheduler_config is not None)
            and (lr_scheduler_config.lr_scheduler_class is not None)):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler,
                                     lr_scheduler_config.lr_scheduler_class)
        if lr_scheduler_class is not None:
            # update the lr_scheduler params from the config file
            _params = (dict() if lr_scheduler_config.params is None
                       else dot_notation_to_dict(lr_scheduler_config.params))

            lr_scheduler = partial(lr_scheduler_class, **_params)
            logger.info(f"lr_scheduler print:\n{lr_scheduler}")

            return lr_scheduler


def init_resume(config, model, model_params, path_logs, logger):
    resume_training = (config.checkpoint if config.checkpoint is not None
                       else DictToDotNotation(dict(resume=False)))
    if resume_training.resume:
        path_old_logs = (Path(__file__).resolve().parents[2].absolute()
                         / "logs" / str(resume_training.date)
                         / resume_training.time)
        if path_old_logs.exists():
            resume_training.path = path_old_logs
        else:
            logger.warning(f"The folder {path_old_logs} does not exist"
                           " .. the checkpoint point will be ignored")
            resume_training.resume = False
    return resume_training


def train(config, model_trainer, use_mc_dropout, train_dataloader,
          val_dataloader, is_train, is_uncertainties, logger):
    logger.info(" ------ Training the model")
    backend_uncertainties = config.training.backend_uncertainties
    if backend_uncertainties is None:
        logger.info("the backend for MC-Dropout was not specified in the "
                    "evaluation section of the config file .. using numpy")
        backend_uncertainties = 'torch'

    uncertainties_per_epoch = config.training.uncertainties_per_epoch
    if uncertainties_per_epoch is None:
        uncertainties_per_epoch = is_uncertainties

    model_trainer.train(train_dataloader, val_dataloader,
                        backend_uncertainties, uncertainties_per_epoch,
                        use_mc_dropout)


def uncertainties(config, model_trainer, test_dataloader, use_mc_dropout,
                  use_latest, path_logs, logger):
    logger.info(" ------ Computing uncertainties")

    mc_forward_passes = (config.evaluation.mc_forward_passes
                         if config.evaluation.mc_forward_passes is not None
                         else 10)
    mean, variance, entropy, cond_entropy, mutual_info = \
        model_trainer.get_uncertainties(
            test_dataloader, mc_forward_passes, backend='numpy',
            use_mc_dropout=use_mc_dropout, use_latest=use_latest)
    logger.info(f"Entropy: mean={entropy.mean():.3e} -"
                f" std={entropy.std():.3e}")
    logger.info(f"Conditional entropy: mean={cond_entropy.mean():.3e} -"
                f" std={cond_entropy.std():.3e}")
    logger.info(f"Mutual information: mean={mutual_info.mean():.3e}"
                f" - std={mutual_info.std():.3e}")

    # save uncertainties arrays
    if config.evaluation.save_uncertainties_metrics:
        path_metrics = path_logs / "uncertainties"
        Path(path_metrics).mkdir(parents=True, exist_ok=True)
        np.save(path_metrics / "mean.npy", mean)
        np.save(path_metrics / "variance.npy", variance)
        np.save(path_metrics / "entropy.npy", entropy)
        np.save(path_metrics / "cond_entropy.npy", cond_entropy)
        np.save(path_metrics / "mutual_info.npy", mutual_info)
        logger.info(f"uncertainty metrics saved at {path_metrics}")


def dirichlet_uncertainties(config, model_trainer, test_dataloader,
                            use_latest, path_logs, logger):
    names = ['expected_entropy', 'distributional_uncertainty',
             'differential_entropy']
    values = model_trainer.dirichlet_uncertainties(
        test_dataloader, backend='numpy', use_latest=use_latest)

    if values is not None:
        logger.info(" ------ Computing Dirichlet uncertainties")
        # save uncertainties arrays
        path_metrics = path_logs / "evidential"
        is_save = bool(config.evaluation.save_evidential_metrics)
        if is_save:
            Path(path_metrics).mkdir(parents=True, exist_ok=True)

        for (name, value) in zip(names, values):
            logger.info(f"{name}: mean={value.mean():.3e} -"
                        f" std={value.std():.3e}")
            if is_save:
                np.save(path_metrics / f"{name}.npy", value)


def save_id_ood_logits(config, id_loader, model_trainer, use_latest,
                       path_logs, device, logger):
    data_section = dot_notation_to_dict(config.data)
    batch_size = data_section.get('batch_size', 512)
    id_dataset_name = data_section.get('dataset', 'mnist')
    ood_dataset_name = id_ood_map.get(id_dataset_name)

    if ood_dataset_name is not None:
        ood_dataset = train_val_test_datasets(
            dataset=ood_dataset_name, id_dataset=id_dataset_name)[2]
        _, _, ood_loader = get_dataloaders(batch_size, None, None,
                                           ood_dataset, device)
    else:
        ood_loader = None

    nbr_forward_passes = config.evaluation.mc_forward_passes
    nbr_forward_passes *= int(bool(config.evaluation.use_mc_dropout))

    folder_logits = path_logs / 'logits'
    folder_logits.mkdir(parents=True, exist_ok=True)
    files_path = (folder_logits / f'id_{id_dataset_name}_samples.npy',
                  folder_logits / f'ood_{ood_dataset_name}_samples.npy')
    for (loader, file_path) in zip((id_loader, ood_loader), files_path):
        # call model_trainer.save_logits(loader, use_latest, path)
        logger.info(f' -> saving logits: {file_path}')
        model_trainer.save_logits(loader, file_path, use_latest,
                                  nbr_forward_passes)


def timer_and_tracker(fct):
    def inner(logger, *args, **kwargs):
        # init CO2 emissions tracker
        try:
            tracker = EmissionsTracker(log_level='error',
                                       logging_logger=logger)
            tracker.start()
        except Exception as e:
            tracker = None
            logger.warning(f'could not track emisions: {e}')

        # start time
        start_time = time.time()

        # set default output to None
        output = None

        try:
            # run the main function
            output = fct(logger, *args, **kwargs)
        except Exception as error:
            logger.exception(f'Abording main. Error raised:\n{error}')
        finally:
            # total duration
            duration_s = int(time.time() - start_time)
            hours, remainder = divmod(duration_s, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration = f'{hours:02d}H {minutes:02d}min {seconds:02d}s'

            # get the emissions
            emissions = tracker.stop() if tracker is not None else None

            if emissions is None:
                # FIXME: why None?
                emissions = 0.
                logger.warning('None value for emissions')

        # logs
        logger.info(f'Duration: The function took {duration}')
        logger.info(f'CO2 emission: The function emitted {emissions:.3e} kgs')

        return output
    return inner


@timer_and_tracker
def main(logger, config, device, datasets, input_shape, nb_classes, path_logs,
         final_dest, is_train, is_test, is_uncertainties, save_idood_logits):
    # Datasets
    assert len(datasets) == 3

    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        config.data.batch_size, *datasets, device=device)

    # Model
    model, model_params, model_class = init_model(
        config, input_shape, nb_classes, device, logger)

    # Loss function
    loss_fct = init_loss(config, model, model_params,
                         len(datasets[0]), logger)

    # Optimizer
    optimizer = init_optimizer(config, logger)

    # Learning rate schedulers
    lr_scheduler = init_lr_scheduler(config, logger)

    # in case we want to resume training
    resume_training = init_resume(config, model, model_params,
                                  path_logs, logger)

    # Trainer object
    model_name = (model_class if model_params['num_models'] == 1
                  else 'EnsembleNet')
    if not isinstance(model_name, str):
        model_name = model_name.__name__
    trainer_params = dot_notation_to_dict(config.training)
    model_trainer = ModelTrainer(model, optimizer, loss_fct, model_name,
                                 model_params, path_logs, resume_training,
                                 lr_scheduler=lr_scheduler, **trainer_params)

    # Train the model
    use_mc_dropout = config.evaluation.use_mc_dropout
    if use_mc_dropout is None:
        use_mc_dropout = True

    if is_train:
        train(config, model_trainer, use_mc_dropout, train_dataloader,
              val_dataloader, is_train, is_uncertainties, logger)

    # Test the model
    use_latest = bool(config.evaluation.use_latest)
    if is_test:
        logger.info(" ------ Testing the model")
        loss, accuracy = model_trainer.test(test_dataloader, use_latest)
        logger.info(f"test_loss={loss:.3e} - test_accuracy={accuracy:.2%}")

    # Compute uncertainties
    if is_uncertainties:
        uncertainties(config, model_trainer, test_dataloader, use_mc_dropout,
                      use_latest, path_logs, logger)
        dirichlet_uncertainties(config, model_trainer, test_dataloader,
                                use_latest, path_logs, logger)

    # Save ID and OOD logits
    if save_idood_logits:
        logger.info(" ------ Saving logits for ID and OOD samples")
        save_id_ood_logits(config, test_dataloader, model_trainer,
                           use_latest, path_logs, device, logger)

    # Move the folder from path_logs to final_dest
    if final_dest is not None:
        logger.info(f"moving the logs folder from {path_logs} to {final_dest}")
        final_dest.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(path_logs, final_dest)
        except Exception as e:
            logger.warn(e)

    return model_trainer


def main_cli():
    # argparser
    parser = argparse.ArgumentParser('main')
    parser.add_argument('--notrain', action='store_true')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--no-uncertainties', action='store_true')
    parser.add_argument('--save-idood-logits', action='store_true')
    parser.add_argument('config_file')
    args = parser.parse_args()

    # config
    config, logger, path_logs, final_dest = init_config(args.config_file)

    # gpu
    if config.device.use_gpu:
        device = get_device()
        logger.info(f"using device={device}")
    else:
        device = torch.device('cpu')
        logger.warning(f"cuda not found. Using {device}")

    # Datasets
    *datasets, input_shape, nb_classes = init_data(config, logger, device)

    # run the main function
    main(logger, config, device, datasets, input_shape, nb_classes,
         path_logs, final_dest, not args.notrain, not args.notest,
         not args.no_uncertainties, args.save_idood_logits)


if __name__ == "__main__":
    raise SystemExit(main_cli())
