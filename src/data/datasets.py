from collections.abc import Iterable
from itertools import cycle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN, EMNIST
from ddu_dirty_mnist import AmbiguousMNIST
import torchvision.transforms as t_transforms
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

from .embeddings import compute_and_save_embeddings
from .. import models as my_models

# logger object
import logging
logger = logging.getLogger('main_all')


SETS_NORMALIZATION = dict(mnist=dict(mean=(0.131,),
                                     std=(0.308,),
                                     set_cls=MNIST,
                                     labels_attr="targets"),
                          # dirtymnist=dict(mean=(-0.0651,),
                          #                 std=(0.8902,),
                          #                 set_cls=DirtyMNIST,
                          #                 labels_attr="targets"),
                          ambiguousmnist=dict(mean=(-0.13,),
                                              std=(0.7574,),
                                              set_cls=AmbiguousMNIST,
                                              labels_attr="targets"),
                          emnist=dict(mean=(0.175,),
                                      std=(0.333,),
                                      set_cls=EMNIST,
                                      labels_attr="targets"),
                          fashionmnist=dict(mean=(0.286,),
                                            std=(0.353,),
                                            set_cls=FashionMNIST,
                                            labels_attr="targets"),
                          cifar10=dict(mean=(0.49, 0.48, 0.445),
                                       std=(0.247, 0.243, 0.2615),
                                       set_cls=CIFAR10,
                                       labels_attr="targets"),
                          svhn=dict(mean=(0.44, 0.445, 0.47),
                                    std=(0.20, 0.20, 0.20),
                                    set_cls=SVHN,
                                    labels_attr="labels"))


def dataset_info(dataset: str):
    dataset = dataset.lower()
    if 'mnist' in dataset:
        if dataset == 'emnist':
            return (1, 28, 28), 47
        return (1, 28, 28), 10
    elif any(v in dataset for v in ('cifar', 'svhn')):
        return (3, 32, 32), 10
    else:
        return None, None


def compute_mean_std_dataset(loader):
    mean, mean_2 = 0., 0.
    n = 0

    for data, _ in loader:
        n += data.shape[0]
        reshaped_data = data.view(*data.shape[:2], -1)
        mean += reshaped_data.mean(dim=2).sum(dim=0)
        mean_2 += (reshaped_data**2).mean(dim=2).sum(dim=0)

    mean /= n
    mean_2 /= n

    return mean, torch.sqrt(mean_2 - mean**2)


def inv_normalizer(dataset=None, mean=None, std=None):
    """
    reverse the normalization transformation
    """
    if dataset is not None and dataset in SETS_NORMALIZATION:
        mean = SETS_NORMALIZATION[dataset]['mean']
        std = SETS_NORMALIZATION[dataset]['std']

    if mean is None or std is None:
        raise ValueError('`dataset` and `(mean, std)` are None')

    return t_transforms.Normalize(mean=[-m/s for (m, s) in zip(mean, std)],
                                  std=[1/s for s in std])


def drop_labels_from_datasets(datasets, d_labels):
    """
    datasets: Iterable of Subset objects
    d_labels: int/list[int] labels to be dropped

    return: Iterable of Subset without the `d_labels`
    """
    out_datasets = []
    if (d_labels is not None
            and isinstance(d_labels, (Iterable, int, float))):
        d_labels = torch.tensor(d_labels)
        for i, subset in enumerate(datasets):
            dataset_ = subset.dataset
            label_attr = ('targets' if 'targets' in vars(dataset_).keys()
                          else 'labels')
            labels = getattr(dataset_, label_attr)
            # avoid warning: UserWarning: To copy construct from a tensor ...
            if isinstance(labels, torch.Tensor):
                labels = labels.clone().detach()[subset.indices]
            else:
                labels = torch.tensor(labels)[subset.indices]
            indices = np.array(subset.indices)[~torch.isin(labels, d_labels)]
            logger.info(f"dataset {i}: dropping {len(labels) - len(indices)}"
                        f" samples from {len(labels)} samples for"
                        f" the label' list: {d_labels.tolist()}")
            out_datasets.append(Subset(dataset_, indices))
        return out_datasets
    else:
        return datasets


def download_sets(path_dataset, dataset):
    """
    script to check if we need to download the datasets.
    The main goal of this script is to avoid the prints when loading
    CIFAR10 and SVHN ...
    A bit slower but more guaranteed not to break the code :)
    """
    dataset = dataset.lower()
    commun_kwargs = dict(root=path_dataset, download=False)
    if dataset in ('cifar10', 'svhn'):
        dataset_ = SETS_NORMALIZATION[dataset]['set_cls']
        try:
            if dataset == "svhn":
                dataset_(split='train', **commun_kwargs)
                dataset_(split='test', **commun_kwargs)
            elif dataset == "emnist":
                if dataset == "emnist":
                    commun_kwargs['split'] = 'balanced'
                dataset_(train=True, **commun_kwargs)
                dataset_(train=False, **commun_kwargs)
        except RuntimeError:
            return True
    return False


def train_val_test_datasets(path_dataset=None, ratio_train_val=0.2,
                            ratio_subset_train=1, dataset="mnist",
                            download=None, id_dataset=None, normalize=True,
                            drop_labels=None, embeddings=None,
                            device=torch.device('cpu'), **kwargs):
    # FIXME: drop samples based on labels only work for now for the
    #        latest labels ... dropping the first label or a label in
    #        the middle will result in an error when training the model

    assert ratio_subset_train != 0
    assert min(0, ratio_train_val, ratio_subset_train) == 0
    assert max(1, ratio_train_val, ratio_subset_train) == 1

    dataset = dataset.lower()

    if path_dataset is None:
        path_dataset = Path(__file__).resolve().parents[2].absolute() / "data"
        logger.info("using library path for the data directory"
                    f" {str(path_dataset)}")

    if dataset not in SETS_NORMALIZATION:
        raise ValueError('not a valid set name')

    dataset_ = SETS_NORMALIZATION[dataset]['set_cls']
    labels_attr = SETS_NORMALIZATION[dataset]['labels_attr']
    mean = SETS_NORMALIZATION[dataset]['mean']
    std = SETS_NORMALIZATION[dataset]['std']

    if id_dataset is not None and isinstance(id_dataset, str):
        id_dataset = id_dataset.lower()
        if id_dataset.lower() in SETS_NORMALIZATION:
            # use id mean and std instead of the dataset values
            mean = SETS_NORMALIZATION[id_dataset]['mean']
            std = SETS_NORMALIZATION[id_dataset]['std']

    if dataset == 'ambiguousmnist':
        data_transform = []
    else:
        data_transform = [t_transforms.ToTensor()]

    if normalize:
        data_transform.append(t_transforms.Normalize(mean, std))
    data_transform = t_transforms.Compose(data_transform)

    commun_kwargs = dict(root=path_dataset, transform=data_transform)
    commun_kwargs['download'] = (download if download is not None
                                 else download_sets(path_dataset, dataset))

    if dataset == "svhn":
        train_val_dataset = dataset_(split='train', **commun_kwargs)
        test_dataset = dataset_(split='test', **commun_kwargs)
    else:
        if dataset == "emnist":
            commun_kwargs['split'] = 'balanced'
        train_val_dataset = dataset_(train=True, **commun_kwargs)
        test_dataset = dataset_(train=False, **commun_kwargs)

    input_shape, nb_classes = dataset_info(dataset)

    # NOTE: embedding section
    if embeddings is not None:
        if embeddings.lower() == "identity":
            encoder = nn.Identity()
            model_name = "identity"
        elif embeddings.lower() == "flatten":
            encoder = nn.Flatten()
            model_name = "flatten"
        else:
            encoder = getattr(my_models, embeddings, my_models.VGG11)(
                input_shape=input_shape, nb_classes=nb_classes,
                pretrained=True, freeze=True, device=device)
            encoder = nn.Sequential(
                *list(next(encoder.children()).children())[:-1],
                nn.Flatten())
            model_name = encoder.__class__.__name__
        embeddings_path = (path_dataset / 'embeddings'
                           / f"{dataset}_{id_dataset}" / model_name)
        (embeddings_path / 'train').mkdir(parents=True, exist_ok=True)
        (embeddings_path / 'test').mkdir(parents=True, exist_ok=True)

        train_val_dataset, _ = compute_and_save_embeddings(
            encoder, model_name, train_val_dataset,
            embeddings_path / 'train', device)
        test_dataset, input_shape = compute_and_save_embeddings(
            encoder, model_name, test_dataset,
            embeddings_path / 'test', device)
        logger.info(f"using embeddings computed using pretrained {model_name}")

        with torch.no_grad():
            torch.cuda.empty_cache()

    # all datasets are of type Subset
    test_dataset = Subset(test_dataset, range(len(test_dataset)))

    # split train_val dataset is done randomly
    train_dataset, val_dataset = random_split(
        train_val_dataset, [1 - ratio_train_val, ratio_train_val])

    # taking a subset of the training dataset
    if ratio_subset_train != 1:
        _labels = getattr(train_dataset.dataset, labels_attr)
        targets = np.array(_labels)[train_dataset.indices]

        train_idx, _ = train_test_split(train_dataset.indices,
                                        train_size=ratio_subset_train,
                                        shuffle=True, stratify=targets)
        # we take train_dataset from train_val_dataset
        # to avoid having a Subset of Subset
        train_dataset = Subset(train_val_dataset, train_idx)

    datasets = train_dataset, val_dataset, test_dataset
    return (*drop_labels_from_datasets(datasets, drop_labels),
            input_shape, nb_classes)


def get_dataloaders(batch_size,
                    train_dataset=None,
                    validation_dataset=None,
                    test_dataset=None,
                    device=torch.device('cpu')):
    '''
    function to return the train, validation and test dataloaders
    based on the batch size.
    for the train dataloader, shuffle is set to True while
    for the validation and test dataloaders is set to False
    if a dataset is None, the returned 'dataloader' is also None
    NB: batch_size is either an int or an iterable of ints
    '''
    kwargs_loader = (dict() if device == torch.device('cpu')
                     else dict(num_workers=1, pin_memory=True))

    datasets = (train_dataset, validation_dataset, test_dataset)
    batch_sizes = ([batch_size for _ in range(len(datasets))]
                   if isinstance(batch_size, int) else cycle(batch_size))

    return tuple(
        None if dataset is None
        else DataLoader(dataset, batch_size=batch_size,
                        shuffle=(i == 0), **kwargs_loader)
        for i, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes)))


class TensorsToDataset(Dataset):
    """
    Convert a list of tensors to a dataset
    """

    def __init__(self, *args):
        super().__init__()
        assert len(args) > 0, 'provide at least one tensor'
        assert all(len(args[0]) == len(x) for x in args), \
            'all the tensors should be of the same size'
        self._args = tuple(args)

    def __len__(self):
        return self._args[0].shape[0]

    def __getitem__(self, index):
        return tuple([x[index] for x in self._args])


if __name__ == '__main__':
    from itertools import chain

    print("-"*50)
    for dataset in ['MNIST', 'AmbiguousMNIST', 'FashionMNIST',
                    'CIFAR10', 'SVHN', 'EMNIST']:
        datasets = train_val_test_datasets(dataset=dataset)
        loaders = get_dataloaders(512, *datasets)
        for (name, loader) in zip(['train', 'val', 'test'], loaders):
            mean, std = compute_mean_std_dataset(loader)
            print(f'{dataset} - {name}: mean={mean} - std={std}')
        mean, std = compute_mean_std_dataset(chain(*loaders[:-1]))
        print(f'{dataset} - train-val: mean={mean} - std={std}')
        mean, std = compute_mean_std_dataset(chain(*loaders))
        print(f'{dataset} - all: mean={mean} - std={std}')
        print("-"*50)
