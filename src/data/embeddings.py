from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@torch.no_grad()
def compute_and_save_embeddings(encoder, model_name, dataset, path,
                                device=None, force=False):
    path = Path(path).resolve().absolute()
    if next(path.iterdir(), None) is not None and not force:
        embeddings_dataset = EmbeddingsDataset(path)
        input_shape = embeddings_dataset.input_shape
        return embeddings_dataset, input_shape

    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device

    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    msg = f"computing embeddings ({model_name})"
    results = zip(*[(encoder(d.to(device)).cpu(), t)
                    for (d, t) in tqdm(loader, desc=msg)])

    for (arr, fname) in zip(results, ('embeddings', 'targets')):
        arr = torch.cat(arr, dim=0)
        if fname == 'embeddings':
            input_shape = tuple(arr[0].shape)
        np.save(path / f'{fname}.npy', arr)

    return EmbeddingsDataset(path), input_shape


class EmbeddingsDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path).resolve().absolute()
        self.embeddings_path = self.path / 'embeddings.npy'
        self.targets = np.load(self.path / 'targets.npy')
        self.labels = self.targets
        self.num_samples = len(self.targets)
        self.mmap_embeddings = np.load(self.embeddings_path, mmap_mode='r+')
        self.input_shape = tuple(self.mmap_embeddings[0].shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.mmap_embeddings[idx], self.targets[idx])
