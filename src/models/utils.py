import torch
from torch.nn.utils.parametrizations import spectral_norm
from pathlib import Path

from .. import models

# logger object
import logging
logger = logging.getLogger('main_all')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_SN(layer, SN):
    return spectral_norm(layer) if SN else layer


def model_loader_from_ckpt(date=None, time=None, ckpt_file=None,
                           device=torch.device('cpu')):
    """
    load the checkpoint saved for a training of the model based on the date
    and the time

    return the best trained model and the latest trained model
    date: int/string format=%Y%m%d
    time: string format=%Hh%Mmin%Ss (old version: %Hh%Mm%Ss)
    """
    # get path to checkpoint
    if ckpt_file is None:
        path_logs = Path(__file__).resolve().parents[2].absolute() / "logs"
        ckpt_file = path_logs / str(date) / time / "checkpoint.pth"
    else:
        ckpt_file = Path(ckpt_file).resolve().absolute()

    if ckpt_file.is_dir():
        ckpt_file = ckpt_file / "checkpoint.pth"

    # load checkpoint
    ckpt = torch.load(ckpt_file, map_location=device)

    # model params
    model_params = ckpt['model_params']
    model_params['device'] = device

    # force pretraind to false
    if 'pretrained' in model_params:
        model_params['pretrained'] = False

    # model class
    try:
        if isinstance(ckpt['model_name'], str):
            model_class = getattr(models, ckpt['model_name'])
        else:
            model_class = getattr(models, ckpt['model_name'].__name__)
    except KeyError:
        model_class = ckpt['model_name']

    # FIXME: for models (vgg for example), change the condition on the kwargs
    if model_params.get('num_models') == 1:
        model_params.pop('num_models')
        model_params.pop('model_class')

    # best model
    b_model = model_class(**model_params)
    logger.info(f"the best model was trained for {ckpt['b_epoch']} epochs")
    b_model.load_state_dict(ckpt['b_model_state'], strict=True)
    b_model.eval()

    # best model
    l_model = model_class(**model_params)
    logger.info(f"the latest model was trained for {ckpt['l_epoch']} epochs")
    l_model.load_state_dict(ckpt['l_model_state'], strict=True)
    l_model.eval()

    return b_model, l_model
