from pathlib import Path
import shutil
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections.abc import Iterable
from functools import partial
from itertools import cycle
from copy import deepcopy

from .utils import train_model, test_model
from ..loss import (EvidentialLoss, SoftplusCrossEntropyLoss)
from ..models.ensemble import EnsembleNet
from ..metrics.uncertainties import (MC_Dropout, ensemble_uncertainties,
                                     evidential_uncertainties)
from ..evaluations.compute_logits import (model_generator, compute_logits,
                                          save_logits)


# logger object
import logging
logger = logging.getLogger('main_all')
logger_file = logging.getLogger('main_file')


class EarlyStopping:
    def __init__(self, tolerance=1, min_delta=0, maximize=True):
        self.tolerance = max(tolerance, 1)
        self.min_delta = min_delta
        self.counter = 0
        self.maximize = maximize
        self.baseline = (1 - 2*int(maximize)) * float('inf')

    def reset(self, value) -> bool:
        if self.maximize:
            return self.baseline < value
        else:
            return self.baseline > value

    def waiting(self, value) -> bool:
        if self.maximize:
            return value + self.min_delta < self.baseline
        else:
            return value > self.baseline + self.min_delta

    def __call__(self, value) -> bool:
        if self.reset(value):
            self.baseline = value
            self.counter = 0

        elif self.waiting(value):
            self.counter += 1
            return self.counter >= self.tolerance

        return False


# FIXME: save state dict for list of optimizers
class ModelCheckpoint:
    def __init__(self, model, optimizer, model_name, model_params, path_logs):
        ''' attributes starting with "b_" are for the best model
        while those starting with "l_" are for the latest model '''
        self.model_name = model_name
        self.model_params = model_params
        # is list of optimizers
        self.list_opts = isinstance(optimizer, Iterable)
        # best model
        self.b_model_state = model.state_dict()
        self.b_optimizer_state = (optimizer.state_dict() if not self.list_opts
                                  else [opt.state_dict() for opt in optimizer])
        self.b_epoch = 1
        self.b_training_loss = float('inf')
        # latest model
        self.l_model_state = model.state_dict()
        self.l_optimizer_state = (optimizer.state_dict() if not self.list_opts
                                  else [opt.state_dict() for opt in optimizer])
        self.l_epoch = 1
        self.l_training_loss = float('inf')
        self.path_logs = path_logs
        # path of the .pth file
        self.path = self.path_logs / "checkpoint.pth"
        # checkpoint elements to be saved
        self.save_list = ['b_epoch', 'b_training_loss', 'b_model_state',
                          'b_optimizer_state', 'l_epoch', 'l_training_loss',
                          'l_model_state', 'l_optimizer_state',
                          'model_name', 'model_params', 'path']

        # start by saving the checkpoint of the initial model
        checkpoint = {k: v for (k, v) in vars(self).items()
                      if k in self.save_list}
        torch.save(checkpoint, self.path)

    def update(self, training_loss, epoch, model_state, optimizer, is_best):
        self.l_training_loss = training_loss
        self.l_epoch = epoch
        self.l_model_state = model_state
        self.l_optimizer_state = (optimizer.state_dict() if not self.list_opts
                                  else [opt.state_dict() for opt in optimizer])

        if is_best:
            self.b_training_loss = training_loss
            self.b_epoch = epoch
            self.b_model_state = deepcopy(model_state)
            self.b_optimizer_state = self.l_optimizer_state
            logger_file.info(
                f" -> best checkpoint saved for epoch: {self.b_epoch}"
                f" - training loss = {self.b_training_loss:.3e}")
        checkpoint = {k: v for (k, v) in vars(self).items()
                      if k in self.save_list}
        torch.save(checkpoint, self.path)

    def update_from_old_checkpoint(self, old_ckpt, path_sw):
        self.l_epoch = old_ckpt['l_epoch']
        self.l_model_state = old_ckpt['l_model_state']
        self.l_training_loss = old_ckpt['l_training_loss']
        self.l_optimizer_state = old_ckpt['l_optimizer_state']
        # copy the previous summary writer
        shutil.copytree(old_ckpt['path'].parent / "summary_writer", path_sw)
        # copy the plots from the previous training
        if (old_ckpt['path'].parent / "plots").exists():
            shutil.copytree(old_ckpt['path'].parent / "plots",
                            self.path_logs/"plots", dirs_exist_ok=True)


class ModelTrainer:
    def __init__(self, model, optimizer, loss, model_name, model_params,
                 path_logs, resume_training, lr_scheduler=None,
                 training_epochs=20, val_every_k_epochs=1, clip_grad_norm=None,
                 val_mc_dropout_samples=5,
                 save_model_state_every_k_epochs=0,
                 early_stopping=None, **kwargs):
        # model
        self.model = model
        # the best model will be updated during training
        self.best_model = model
        # optimizer(s)
        if isinstance(model, EnsembleNet):
            self.optimizers = [optimizer(m.parameters())
                               for m in model.models]
        else:
            self.optimizers = [optimizer(model.parameters())]

        # learning rate scheduler(s)
        self.lr_schedulers = (None if lr_scheduler is None else
                              [lr_scheduler(opt) for opt in self.optimizers])
        # loss function(s)
        self.loss_fct = loss if isinstance(loss, Iterable) else [loss]
        self.use_softmax = not isinstance(self.loss_fct[0],
                                          SoftplusCrossEntropyLoss)
        self.loss_fct_eval = (nn.CrossEntropyLoss() if self.use_softmax
                              else SoftplusCrossEntropyLoss())

        self.model_out_dim = model.output_dim
        self.device = model.device
        self.training_epochs = training_epochs
        self.total_training_epochs = training_epochs
        self.begin_epoch = 1
        self.end_epoch = training_epochs + 1
        self.path_logs = path_logs
        self.clip_grad_norm = clip_grad_norm

        # early stopping
        if early_stopping is None or not isinstance(early_stopping, dict):
            self.early_stopping = None
            self.validation_frequency = val_every_k_epochs
        else:
            self.early_stopping = EarlyStopping(**early_stopping)
            self.validation_frequency = 1

        # validation
        self.val_mc_dropout_samples = val_mc_dropout_samples

        self.checkpoint = ModelCheckpoint(model, self.optimizers, model_name,
                                          model_params, self.path_logs)
        # summary writer
        self.path_sw = self.path_logs / "summary_writer"

        if resume_training.resume:
            self.resume_from_ckpt(resume_training)

        # summary writer
        Path(self.path_sw).mkdir(parents=True, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.path_sw)

        # Data parallel
        nb_gpus = torch.cuda.device_count()
        if nb_gpus > 0:
            logger.info(f"Using {nb_gpus} GPU{'s' if nb_gpus>1 else ''}")
        else:
            logger.info("Using CPU")
        logger.info("model to nn.DataParallel")
        self.model = nn.DataParallel(self.model)

        # save model state dict every k epochs
        self.init_model_saver(save_model_state_every_k_epochs)

    def resume_from_ckpt(self, resume_training):
        path_old_pth = resume_training.path / "checkpoint.pth"
        if path_old_pth.is_file():
            previous_pth = torch.load(path_old_pth)
            if (previous_pth['model_params'] != self.checkpoint.model_params or
                    previous_pth['model_name'] != self.checkpoint.model_name):
                logger.warning(f"the checkpoint saved at {path_old_pth}"
                               " was trained on a different model and thus"
                               " it won't be used")
            else:
                # logs
                _old_log = (resume_training.path/'logs.log').read_text()
                txt = (f"\n(start) old logs file {resume_training.path}/"
                       "logs.log\n" + _old_log + "(end) old logs"
                       f" file {resume_training.path}/logs.log\n")
                logger.info(f"old logs from the previous training:\n{txt}")

                logger.info("resuming the training of the model"
                            f" for additional {self.training_epochs} epochs")
                # update checkpoint
                self.total_training_epochs += previous_pth['l_epoch']
                self.begin_epoch += previous_pth['l_epoch']
                self.end_epoch += previous_pth['l_epoch']
                self.model.load_state_dict(previous_pth['l_model_state'])
                if isinstance(previous_pth['l_optimizer_state'], Iterable):
                    for opt, state in zip(self.optimizers,
                                          previous_pth['l_optimizer_state']):
                        opt.load_state_dict(state)
                else:
                    self.optimizers.load_state_dict(
                        previous_pth['l_optimizer_state'])
                previous_pth['path'] = path_old_pth
                self.checkpoint.update_from_old_checkpoint(previous_pth,
                                                           self.path_sw)
        else:
            logger.warning(f"the file {path_old_pth} does not exist"
                           " .. The model will be trained from scratch")

    def init_model_saver(self, save_model_state_every_k_epochs):
        if save_model_state_every_k_epochs > 0:
            logger.info('the state dict of the model will be saved every'
                        f' {save_model_state_every_k_epochs} epochs')
            self.path_state_dict = self.path_logs / 'models_state_dict'
            Path(self.path_state_dict).mkdir(parents=True, exist_ok=True)
            # save the initialized model (epoch=0)
            torch.save(self.model.module.state_dict(),
                       self.path_state_dict / 'epoch_0.pth')

            self.model_state_k_epochs = lambda e: not bool(
                e % save_model_state_every_k_epochs)
        else:
            self.model_state_k_epochs = lambda e: False

    def train_epoch(self, train_loader, epoch):
        """
        train the model for epoch `epoch`
        return the loss and the accuracy
        in case an ensemble is used, the mean loss and accuracy are returned
        """
        if isinstance(self.model.module, EnsembleNet):
            model_list = self.model.module.models
            iter = tqdm(zip(model_list, self.optimizers, cycle(self.loss_fct)),
                        desc='Ensemble', total=len(model_list), leave=False)
        else:
            model_list = [self.model.module]
            iter = zip(model_list, self.optimizers, cycle(self.loss_fct))

        nb_samples = len(train_loader.dataset)
        # TODO: save loss and accuracy in list of values
        #       and report the mean
        loss_epoch = 0
        acc_epoch = 0

        # NOTE: starting with the model loop before the data loop
        # will have no to little effect when using GPUs
        # and the difference is more noticed on CPUs
        for (model, optimizer, loss_fct) in iter:
            if isinstance(loss_fct, EvidentialLoss):
                loss_fct = partial(loss_fct, epoch=epoch)

            loss_model, acc_model = train_model(
                model, train_loader, loss_fct, optimizer, nb_samples,
                self.clip_grad_norm, self.device)

            acc_epoch += acc_model
            loss_epoch += loss_model

        return loss_epoch/len(model_list), acc_epoch/len(model_list)

    def validation_epoch(self, loss_epoch, epoch, validation_loader,
                         len_val_dataset, backend_uncertainties,
                         uncertainties_per_epoch, use_mc_dropout):
        is_best = loss_epoch < self.checkpoint.b_training_loss
        is_break = False
        # it will be changed if validation loader is not None ...
        loss_val = loss_epoch

        if validation_loader is not None:
            if self.early_stopping is not None:
                loss_val, acc_val = self.model_validation(
                    epoch, validation_loader, len_val_dataset,
                    backend_uncertainties, uncertainties_per_epoch,
                    use_mc_dropout)

                _baseline = self.early_stopping.baseline
                if self.early_stopping.maximize:
                    is_break = self.early_stopping(acc_val)
                    is_best = acc_val > _baseline
                else:
                    is_break = self.early_stopping(loss_val)
                    is_best = loss_val < _baseline

                is_break = is_break or (acc_val == 1)

                if is_break:
                    _msg = (f'[EARLY STOPPING] stopped at {epoch=} - '
                            f'(best epoch at {self.checkpoint.b_epoch}; '
                            f'tolerance={self.early_stopping.tolerance}; '
                            f'min_delta={self.early_stopping.min_delta})')
                    logger.info(_msg)

            elif epoch % self.validation_frequency == 0:
                loss_val, _ = self.model_validation(
                    epoch, validation_loader, len_val_dataset,
                    backend_uncertainties, uncertainties_per_epoch,
                    use_mc_dropout)

        return (is_best, is_break, loss_val)

    def train(self, train_loader, validation_loader=None,
              backend_uncertainties='torch', uncertainties_per_epoch=False,
              use_mc_dropout=True):
        logger.info(f"training the model for {self.training_epochs} epochs")

        len_val_dataset = len(validation_loader.dataset)

        for epoch in tqdm(range(self.begin_epoch, self.end_epoch),
                          desc='training loop'):
            loss_epoch, acc_epoch = self.train_epoch(train_loader, epoch)
            self.summary_writer.add_scalar('metrics/train_loss',
                                           loss_epoch, epoch)
            self.summary_writer.add_scalar('metrics/train_acc',
                                           acc_epoch, epoch)
            # check if loss is NaN
            if loss_epoch != loss_epoch:
                logger.warning('Stopping the training due to NaN '
                               f'loss at epoch {epoch} - using the '
                               'saved model from the previous epoch')
                _ckpt = torch.load(self.checkpoint.path,
                                   map_location=self.device)
                self.best_model.module.load_state_dict(_ckpt['b_model_state'])
                self.model.module.load_state_dict(_ckpt['l_model_state'])
                break

            # validation
            is_best, is_break, loss_val = self.validation_epoch(
                loss_epoch, epoch, validation_loader, len_val_dataset,
                backend_uncertainties, uncertainties_per_epoch, use_mc_dropout)

            if is_break:
                break

            # step learning rate scheduler(s)
            if self.lr_schedulers is not None:
                for (i, lr_scheduler) in enumerate(self.lr_schedulers, 1):
                    if isinstance(lr_scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(loss_val)
                    else:
                        lr_scheduler.step()
                    _lr = lr_scheduler.optimizer.param_groups[0]["lr"]
                    self.summary_writer.add_scalar(f'learning_rates/opt_{i}',
                                                   _lr, epoch)

            # update the model checkpoint
            if is_best:
                # update the best model
                self.best_model = self.model

            m_state_dict = self.model.module.state_dict()
            # TODO: include lr_schedulers in the checkpoint ...
            self.checkpoint.update(loss_epoch, epoch, m_state_dict,
                                   self.optimizers, is_best)

            # save model state dict
            if self.model_state_k_epochs(epoch):
                torch.save(self.model.module.state_dict(),
                           self.path_state_dict / f'epoch_{epoch}.pth')

            # uncertainties
            if uncertainties_per_epoch:
                uncertainties = self.get_uncertainties(
                    train_loader, self.val_mc_dropout_samples,
                    backend_uncertainties, use_mc_dropout)[2:]

                metrics = dict(entropy=uncertainties[0],
                               conditional_entropy=uncertainties[1],
                               mutual_information=uncertainties[2])

                for name, metric in metrics.items():
                    self.summary_writer.add_scalar(
                        f'train_uncertainties/mean_{name}',
                        metric.mean(), epoch)
                    self.summary_writer.add_scalar(
                        f'train_uncertainties/std_{name}',
                        metric.std(), epoch)

    def model_validation(self, epoch, validation_loader, len_val_dataset,
                         backend_uncertainties, uncertainties_per_epoch,
                         use_mc_dropout):
        loss_val, acc_val = self.test(validation_loader, use_latest=True,
                                      epoch=epoch)
        self.summary_writer.add_scalar('metrics/val_loss', loss_val, epoch)
        self.summary_writer.add_scalar('metrics/val_acc', acc_val, epoch)

        # compute uncertainties
        if uncertainties_per_epoch:
            uncertainties = self.get_uncertainties(
                validation_loader, self.val_mc_dropout_samples,
                backend_uncertainties, use_mc_dropout)[2:]

            metrics = dict(entropy=uncertainties[0],
                           conditional_entropy=uncertainties[1],
                           mutual_information=uncertainties[2])

            for name, metric in metrics.items():
                self.summary_writer.add_scalar(
                    f'val_uncertainties/mean_{name}', metric.mean(), epoch)
                self.summary_writer.add_scalar(
                    f'val_uncertainties/std_{name}', metric.std(), epoch)
        return loss_val, acc_val

    def test(self, loader, use_latest=False, epoch=None):
        """
        loader: dataloader
        use_latest: bool to determin whather the latest or the best model
                    is to be used for the evaluation
        """
        logger_file.info("evaluating the model with torch.no_grad()")
        _name = 'latest' if use_latest else 'best'
        logger_file.info(f"using the {_name} model for evaluations")

        model = self.model
        if not use_latest:
            model.module.load_state_dict(self.checkpoint.b_model_state)

        loss, accuracy = test_model(model, loader, self.loss_fct_eval,
                                    self.device, len(loader.dataset))
        at_msg = '' if epoch is None or not epoch else f' at epoch {epoch}'
        logger_file.info(f"evaluation{at_msg}: {loss=:.3e} - {accuracy=:.2%}")
        # load the latest checkpoint
        model.module.load_state_dict(self.checkpoint.l_model_state)
        return loss, accuracy

    def save_logits(self, loader, file_path, use_latest=False,
                    nbr_forward_passes=None):
        if loader is None:
            return

        model = self.model
        if not use_latest:
            model.module.load_state_dict(self.checkpoint.b_model_state)

        use_mc_dropout = (nbr_forward_passes > 1)
        model_gen = model_generator(model, nbr_forward_passes, use_mc_dropout)
        logits = [compute_logits(m, loader, self.device) for m in model_gen]
        save_logits(logits, file_path)
        # load the latest checkpoint
        model.module.load_state_dict(self.checkpoint.l_model_state)

    def get_uncertainties(self, loader, nbr_forward_passes, backend='numpy',
                          use_mc_dropout=True, use_latest=False):
        _name = 'latest' if use_latest else 'best'
        logger_file.info(f"using the {_name} model for evaluations")

        model = self.model
        if not use_latest:
            model.module.load_state_dict(self.checkpoint.b_model_state)

        if not (use_mc_dropout or isinstance(self.model.module, EnsembleNet)):
            logger_file.warning("The model is not an ensemble. Using "
                                "MC-Dropout to compute the uncertainties")
            use_mc_dropout = True

        if use_mc_dropout:
            logger_file.info(f"performing MC-dropout with {nbr_forward_passes}"
                             " forward passes")
            values = MC_Dropout(model, nbr_forward_passes, loader,
                                self.model_out_dim, self.use_softmax, backend)
        else:
            logger_file.info("computing uncertainties based on the ensemble")
            values = ensemble_uncertainties(model, loader, backend,
                                            use_softmax=self.use_softmax)
        # load the latest checkpoint
        model.module.load_state_dict(self.checkpoint.l_model_state)
        return values

    def dirichlet_uncertainties(self, loader, backend='numpy',
                                use_latest=False):
        if not isinstance(self.loss_fct[0], EvidentialLoss):
            logger.warn("this method is to be used mainly for models"
                        " trained with EvidentialLoss. None will be returned")
            return None

        toprobs_kwargs = self.loss_fct[0].toprobs_kwargs

        _name = 'latest' if use_latest else 'best'
        logger_file.info(f"using the {_name} model for evaluations")

        model = self.model
        if not use_latest:
            model.module.load_state_dict(self.checkpoint.b_model_state)

        values = evidential_uncertainties(model, loader, backend=backend,
                                          **toprobs_kwargs)

        # load the latest checkpoint
        model.module.load_state_dict(self.checkpoint.l_model_state)
        return values
