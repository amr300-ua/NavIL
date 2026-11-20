from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

CONTRACT_VERSION = '1.0'
SCAN_SIZE = 90
GOAL_SIZE = 2
INPUT_SIZE = SCAN_SIZE + GOAL_SIZE
OUTPUT_SIZE = 2  # (v, w)

REQUIRED_KEYS = ('scan', 'goal', 'v', 'w', 'episode_id')


@dataclass
class ContractConfig:
    """Parámetros de normalización y límites. Se guardan dentro del checkpoint."""

    scan_size: int = SCAN_SIZE
    scan_max_range: float = 12.0
    goal_max_dist: float = 10.0
    v_max: float = 0.30
    w_max: float = 1.50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> 'ContractConfig':
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in fields if k in d})


def scan_angles(scan_size: int = SCAN_SIZE) -> np.ndarray:
    """Ángulo (rad, en [-pi, pi)) de cada rayo según la convención del contrato."""
    a = np.arange(scan_size) * (2.0 * math.pi / scan_size)
    return wrap_angle(a)


def wrap_angle(a):
    """Normaliza ángulos a [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def sanitize_scan(scan, cfg: ContractConfig) -> np.ndarray:
    """Valida y limpia LiDAR. Acepta forma (90,) o (N, 90)."""
    s = np.asarray(scan, dtype=np.float32)
    if s.shape[-1] != cfg.scan_size:
        raise ValueError(
            f'scan debe tener {cfg.scan_size} valores, recibido forma {s.shape}')
    s = np.nan_to_num(s, nan=cfg.scan_max_range, posinf=cfg.scan_max_range,
                      neginf=0.0)
    return np.clip(s, 0.0, cfg.scan_max_range)


def sanitize_goal(goal) -> np.ndarray:
    """Valida el objetivo [distancia, ángulo]. Acepta forma (2,) o (N, 2)."""
    g = np.asarray(goal, dtype=np.float32)
    if g.shape[-1] != GOAL_SIZE:
        raise ValueError(f'goal debe ser [distancia, angulo], recibido forma {g.shape}')
    if not np.all(np.isfinite(g)):
        raise ValueError('goal contiene NaN o inf')
    g = g.copy()
    g[..., 0] = np.maximum(g[..., 0], 0.0)
    g[..., 1] = wrap_angle(g[..., 1])
    return g


def build_features(scan, goal, cfg: ContractConfig) -> np.ndarray:
    """Convierte (scan, goal) en el vector normalizado de entrada a la red.

    scan / max_range -> [0, 1]; dist / goal_max_dist -> [0, 1];
    ángulo / pi -> [-1, 1]. Devuelve float32 de forma (..., 92).
    """
    s = sanitize_scan(scan, cfg) / cfg.scan_max_range
    g = sanitize_goal(goal)
    d = np.clip(g[..., 0:1] / cfg.goal_max_dist, 0.0, 1.0)
    a = g[..., 1:2] / math.pi
    return np.concatenate([s, d, a], axis=-1).astype(np.float32)


def observation_to_features(observation: Mapping, cfg: ContractConfig) -> np.ndarray:
    """Valida un diccionario de observación y devuelve el vector (92,)."""
    for key in ('scan', 'goal'):
        if key not in observation:
            raise KeyError(f"La observación debe contener '{key}'")
    return build_features(observation['scan'], observation['goal'], cfg)


def normalize_targets(v, w, cfg: ContractConfig) -> np.ndarray:
    """(v, w) físicos -> objetivo normalizado en [-1, 1] para la red."""
    t = np.stack([np.asarray(v) / cfg.v_max, np.asarray(w) / cfg.w_max], axis=-1)
    return np.clip(t, -1.0, 1.0).astype(np.float32)


def denormalize_targets(t, cfg: ContractConfig):
    """Salida normalizada de la red -> (v, w) físicos."""
    t = np.asarray(t)
    return t[..., 0] * cfg.v_max, t[..., 1] * cfg.w_max


def validate_dataset(data: Mapping[str, np.ndarray], cfg: ContractConfig) -> int:
    """Comprueba claves y formas del dataset. Devuelve N."""
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f'Faltan claves en el dataset: {missing}')
    n = len(data['v'])
    expected: Sequence = (
        ('scan', (n, cfg.scan_size)), ('goal', (n, GOAL_SIZE)),
        ('w', (n,)), ('episode_id', (n,)),
    )
    for key, shape in expected:
        if data[key].shape != shape:
            raise ValueError(f'{key}: forma {data[key].shape}, esperada {shape}')
    if n == 0:
        raise ValueError('Dataset vacío')
    return n
