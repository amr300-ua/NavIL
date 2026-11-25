import argparse
import time
from typing import Mapping, Optional, Tuple

import numpy as np
import torch

from il_policy.contract import (CONTRACT_VERSION, ContractConfig, build_features,
                                denormalize_targets, observation_to_features)
from il_policy.model import MLPPolicy, build_model


class ILPolicy:
    """Envuelve modelo + normalización + límites detrás de ``predict``."""

    def __init__(self, model: torch.nn.Module, cfg: ContractConfig,
                 device: str = 'cpu', meta: Optional[dict] = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.meta = meta or {}

    # ------------------------------------------------------------------ carga
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ILPolicy':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        version = ckpt.get('contract_version')
        if version != CONTRACT_VERSION:
            raise RuntimeError(
                f'Checkpoint con contrato {version}, el código espera {CONTRACT_VERSION}')
        model = build_model(ckpt['model_config'])
        model.load_state_dict(ckpt['model_state'])
        cfg = ContractConfig.from_dict(ckpt['contract'])
        meta = {k: ckpt.get(k) for k in ('metrics', 'train_args', 'created')}
        return cls(model, cfg, device, meta)

    @classmethod
    def untrained(cls, cfg: Optional[ContractConfig] = None, seed: int = 0) -> 'ILPolicy':
        torch.manual_seed(seed)
        return cls(MLPPolicy(), cfg or ContractConfig())

    # -------------------------------------------------------------- inferencia
    @torch.no_grad()
    def predict(self, observation: Mapping) -> Tuple[float, float]:
        """observation = {'scan': 90 floats, 'goal': [dist, ang]} -> (v, w)."""
        x = observation_to_features(observation, self.cfg)
        out = self.model(torch.from_numpy(x).unsqueeze(0).to(self.device))
        v, w = denormalize_targets(out[0].cpu().numpy(), self.cfg)
        return self._clip(float(v), float(w))

    @torch.no_grad()
    def predict_batch(self, scan: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Versión vectorizada: scan (N, 90), goal (N, 2) -> v (N,), w (N,)."""
        x = torch.from_numpy(build_features(scan, goal, self.cfg)).to(self.device)
        v, w = denormalize_targets(self.model(x).cpu().numpy(), self.cfg)
        return (np.clip(v, -self.cfg.v_max, self.cfg.v_max),
                np.clip(w, -self.cfg.w_max, self.cfg.w_max))

    def _clip(self, v: float, w: float) -> Tuple[float, float]:
        if not (np.isfinite(v) and np.isfinite(w)):
            return 0.0, 0.0  # nunca devolver NaN al robot
        return (float(np.clip(v, -self.cfg.v_max, self.cfg.v_max)),
                float(np.clip(w, -self.cfg.w_max, self.cfg.w_max)))


_DEFAULT: Optional[ILPolicy] = None


def predict(observation: Mapping) -> Tuple[float, float]:
    """Función con la firma exacta del contrato. Usa un modelo sin entrenar
    salvo que antes se llame a ``set_default_policy``."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ILPolicy.untrained()
    return _DEFAULT.predict(observation)


def set_default_policy(policy: ILPolicy) -> None:
    global _DEFAULT
    _DEFAULT = policy


def main(argv=None):
    """Demo: carga un checkpoint y predice sobre observaciones de ejemplo."""
    p = argparse.ArgumentParser(description='Demo de inferencia de il_policy')
    p.add_argument('--model', default=None, help='checkpoint .pt (si no, modelo sin entrenar)')
    p.add_argument('--n', type=int, default=1000, help='repeticiones para medir latencia')
    args = p.parse_args(argv)

    policy = ILPolicy.load(args.model) if args.model else ILPolicy.untrained()
    cases = {
        'pasillo libre, objetivo delante': {'scan': [5.0] * 90, 'goal': [3.0, 0.0]},
        'objetivo a la izquierda': {'scan': [5.0] * 90, 'goal': [2.0, 1.2]},
        'objetivo a la derecha': {'scan': [5.0] * 90, 'goal': [2.0, -1.2]},
        'pared delante': {'scan': [0.35] * 10 + [5.0] * 71 + [0.35] * 9, 'goal': [3.0, 0.0]},
        'LiDAR con inf/NaN': {'scan': [float('inf')] * 45 + [float('nan')] * 45, 'goal': [1.0, 0.0]},
    }
    for name, obs in cases.items():
        v, w = policy.predict(obs)
        print(f'{name:35s} -> v={v:+.3f} m/s  w={w:+.3f} rad/s')

    obs = cases['pasillo libre, objetivo delante']
    t0 = time.perf_counter()
    for _ in range(args.n):
        policy.predict(obs)
    dt = (time.perf_counter() - t0) / args.n * 1e3
    print(f'Latencia media de predict(): {dt:.3f} ms')


if __name__ == '__main__':
    main()
