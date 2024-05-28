from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path


class EnsembleNet(nn.Module):
    def __new__(cls, num_models=None, model_class=None,
                list_date_time=None, **config):
        if num_models is not None and model_class is not None:
            if num_models == 1:
                # simply return the model without the ensemble
                return model_class(**config)
        return object.__new__(cls)

    def __init__(self, num_models=None, model_class=None,
                 list_date_time=None, **config):
        """
        This class can be used in 2 modes:
        * mode 1: ensemble pre-trained models
        * mode 2: initialize a list of the same model with the same config
                  -> this list is to be trained
        -------
        Params:
        * mode 1:
            - list_date_time: iterable of lists/tuples:
                (date, time, ckpt_file (optional))
                -> if None: mode 2
        * mode 2:
            - num_models: (int) number of models for the ensemble
            - model_class: class of the model to be used
            - config: dictionary of the config for the model
        """
        # TODO: the inference of the ensemble model is done in cpu
        # idea1: set device to gpu
        # idea2: use joblib to compute the prediction of each model in parallel
        from .utils import model_loader_from_ckpt

        super().__init__()
        self.list_date_time = list_date_time
        if self.list_date_time is not None:
            # mode 1
            self.best_models = []
            self.latest_models = []
            for element in self.list_date_time:
                # could be a tuple of (date, time) or (date, time, ckpt_file)
                b_model, l_model = model_loader_from_ckpt(*element)
                self.best_models.append(b_model)
                self.latest_models.append(l_model)

            self.best_models = nn.ModuleList(self.best_models)
            self.latest_models = nn.ModuleList(self.latest_models)

            # models = best models
            self.models = self.best_models[:]
            self.num_models = len(self.list_date_time)
            self.path_logs = (Path(__file__).resolve().parents[2].absolute()
                              / "logs")
        elif num_models is not None and model_class is not None:
            # mode 2
            self.config = config
            self.model_class = model_class
            self.num_models = num_models
            self.models = nn.ModuleList([self.model_class(**self.config)
                                         for _ in range(self.num_models)])
        else:
            raise Exception("EnsembleNet should at least be run in one "
                            "of the 2 modes")

    def __getattribute__(self, item):
        if item == 'models':
            return super().__getattr__('models')

        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            # get attributes from the model_class object
            try:
                return super().__getattr__('models')[0].__dict__[item]
            except KeyError:
                return AttributeError(f"EnsembleNet has no attribute {item}")

    def at_epoch(self, epoch):
        if self.list_date_time is not None:
            # first check if all the config files exist
            list_fname = []
            for date, time in self.list_date_time:
                fname = self.path_logs / str(date) / time \
                    / 'models_state_dict' / f'epoch_{epoch}.pth'
                if not fname.is_file():
                    print(f"no checkpoint found for epoch: {epoch};",
                          f"date: {date}; time: {time}")
                    break
                else:
                    list_fname.append(fname)

            if len(list_fname) == len(self.list_date_time):
                for i, (fname, model) in enumerate(
                        zip(list_fname, self.models)):
                    _ckpt = torch.load(fname, map_location=torch.device('cpu'))
                    model.load_state_dict(_ckpt)
                    self.models[i] = model

    def forward(self, x, return_mean=True, mean_logits=False):
        return EnsembleWrapper(self.models)(x, return_mean, mean_logits)


class EnsembleWrapper(nn.Module):
    """
    This wrapper can be used in 2 ways (see `forward` function):
    * if we have ONE input ([BS, dim1, ...]):
        -> compute the output of this input by all models
    * if the number of inputs ([BS, n_models, dim1, ...]) is equal to
      the number of models:
        -> each model in the ensemble will compute the output
           of the associate input
    """
    def __new__(cls, models, return_mean=True, mean_logits=False):
        if isinstance(models, Iterable):
            return object.__new__(cls)
        else:
            if isinstance(models, nn.Module):
                return models
            else:
                raise ValueError('`models` should either be a nn.Module'
                                 ' or a list of nn.Module')

    def __init__(self, models, return_mean=True, mean_logits=False):
        super().__init__()
        assert isinstance(models, Iterable), \
            "EnsembleWrapper takes as input a list of models or a ModuleList"
        self.models = nn.ModuleList(models)

        self.return_mean = return_mean
        self.mean_logits = mean_logits

    def forward(self, x, return_mean=None, mean_logits=None):
        if return_mean is None:
            return_mean = self.return_mean

        if mean_logits is None:
            mean_logits = self.mean_logits

        multiple_inputs = (
            len(self.models) == x.shape[1] and
            vars(self.models[0]).get('input_shape') == x.shape[2:])

        x = (torch.transpose(x, 0, 1) if multiple_inputs
             else [x for _ in range(len(self.models))])

        # shape y: [BS, n models, n classes]
        y = torch.stack(
            [model(data) for (data, model) in zip(x, self.models)], dim=1)

        if return_mean:
            if mean_logits:
                # FIXME: check if this is the right syntax for the logits...
                y = F.softmax(y, dim=2).mean(dim=1).logit()
            else:
                y = y.mean(dim=1)
        return y
