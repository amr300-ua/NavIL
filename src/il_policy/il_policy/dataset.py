"""Carga de datasets y separación train/val/test POR EPISODIOS."""

import glob
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from il_policy.contract import (REQUIRED_KEYS, ContractConfig, build_features,
                                normalize_targets, validate_dataset)

PathLike = Union[str, os.PathLike]


def load_npz(paths: Union[PathLike, List[PathLike]], cfg: ContractConfig) -> Dict[str, np.ndarray]:
    """Carga uno o varios .npz (o un directorio con .npz) y los concatena.

    Los ``episode_id`` se desplazan por fichero para que no colisionen.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files: List[str] = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.npz')))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f'No se encontraron .npz en {paths}')

    parts = {k: [] for k in REQUIRED_KEYS}
    offset = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in REQUIRED_KEYS if k in z.files}
        validate_dataset(d, cfg)
        ep = d['episode_id'].astype(np.int64)
        _, ep = np.unique(ep, return_inverse=True)  # re-indexa 0..E-1
        d['episode_id'] = ep + offset
        offset += int(ep.max()) + 1
        for k in REQUIRED_KEYS:
            parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def split_by_episode(episode_id: np.ndarray, val_frac: float, test_frac: float,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve índices (train, val, test) sin mezclar muestras de un mismo recorrido."""
    episodes = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n = len(episodes)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_test = max(n_test, 1) if test_frac > 0 else 0
        n_val = max(n_val, 1) if val_frac > 0 else 0
    if n - n_test - n_val < 1:
        raise ValueError(f'Muy pocos episodios ({n}) para separar train/val/test')
    test_ep = episodes[:n_test]
    val_ep = episodes[n_test:n_test + n_val]
    train_ep = episodes[n_test + n_val:]

    def idx(eps):
        return np.flatnonzero(np.isin(episode_id, eps))
    return idx(train_ep), idx(val_ep), idx(test_ep)


class ILDataset(Dataset):
    """Dataset de PyTorch con entradas y objetivos ya normalizados."""

    def __init__(self, data: Dict[str, np.ndarray], indices: np.ndarray, cfg: ContractConfig):
        self.x = torch.from_numpy(build_features(data['scan'][indices], data['goal'][indices], cfg))
        self.y = torch.from_numpy(normalize_targets(data['v'][indices], data['w'][indices], cfg))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
