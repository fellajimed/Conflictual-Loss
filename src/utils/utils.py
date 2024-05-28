from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel
import numpy as np
import random
import torch
from functools import reduce
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

# logger object
import logging
logger = logging.getLogger('main_all')


class ProgressParallel(Parallel):
    """
    Parallel object with tqdm progress bar adjusted corrrectly
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def setup_seed(seed=42):
    """ fix random seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"fixing the random seed: {seed}")


def equal_dicts(d1, d2, ignore_keys=[]):
    """
    compare dictionaries d1 and d2 without the keys in ignore_keys
    """
    k_d1 = set(d1).difference(ignore_keys)
    k_d2 = set(d2).difference(ignore_keys)
    return k_d1 == k_d2 and all(d1[k] == d2[k] for k in k_d1)


def rgetattr(obj, attr, *args):
    """
    recursive getattr
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    """
    recursive setattr
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_device():
    """ function to return the available device """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_scalars_from_tensorboad(summary_dir=None, date=None, time=None):
    """
    read the data in the summary writer and return them

    Args:
        summary_dir: path to summary writer directory
        date: str or int for the date for the day of the log
        time: str for the time of the log
    Returns:
        dict[name, steps_values]:
            names: list of the names of the scalars
            steps_values: list of tuples in the same order as 'names'
                          (list of steps, list of values)
    """
    if summary_dir is None:
        if date is not None and time is not None:
            summary_dir = (Path(__file__).resolve().parents[2].absolute()
                           / "logs" / str(date) / time / 'summary_writer')
        else:
            raise ValueError("either 'summary_dir' or '(date, time)'"
                             " should not be None")
    else:
        summary_dir = Path(summary_dir)

    # check if folder exist
    if not summary_dir.is_dir():
        raise ValueError(f"folder '{summary_dir}' does not exist")

    # access the scalars from the summary writer
    event_accumulator = EventAccumulator(str(summary_dir)).Reload()
    scalars = event_accumulator.Tags()['scalars']

    # dictionary to prettify scalar name
    replace_table = {'/': ': ', 'val': 'validation',
                     'acc': 'accuracy', '_': ' '}

    # store results in a dict
    results = dict()

    for scalar in scalars:
        steps = []
        values = []
        # access indices (steps) and values for this scalar
        for scalar_event in event_accumulator.Scalars(scalar):
            steps.append(scalar_event.step)
            values.append(scalar_event.value)
        # pretiffy name
        for old, new in replace_table.items():
            scalar = scalar.replace(old, new)
        results[scalar] = (steps, values)
    return results
