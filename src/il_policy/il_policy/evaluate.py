import argparse
import json
import time

import numpy as np

from il_policy.dataset import load_npz, split_by_episode
from il_policy.policy import ILPolicy


def _r2(y, yhat) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def compute_metrics(policy: ILPolicy, data: dict, indices: np.ndarray) -> dict:
    """Métricas en unidades físicas sobre el subconjunto ``indices``."""
    v_true, w_true = data['v'][indices], data['w'][indices]
    v_pred, w_pred = policy.predict_batch(data['scan'][indices], data['goal'][indices])
    m = {'n_samples': int(len(indices))}
    for name, y, yhat in (('v', v_true, v_pred), ('w', w_true, w_pred)):
        m[f'mae_{name}'] = float(np.mean(np.abs(y - yhat)))
        m[f'rmse_{name}'] = float(np.sqrt(np.mean((y - yhat) ** 2)))
        m[f'r2_{name}'] = _r2(y, yhat)
        # Referencia trivial: predecir siempre la media -> el modelo debe mejorarla
        m[f'baseline_mae_{name}'] = float(np.mean(np.abs(y - y.mean())))
    return m


def measure_latency(policy: ILPolicy, n: int = 500) -> dict:
    obs = {'scan': np.full(policy.cfg.scan_size, 3.0, np.float32), 'goal': [2.0, 0.3]}
    for _ in range(20):
        policy.predict(obs)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy.predict(obs)
        times.append((time.perf_counter() - t0) * 1e3)
    return {'latency_ms_mean': float(np.mean(times)),
            'latency_ms_p95': float(np.percentile(times, 95))}


def main(argv=None):
    p = argparse.ArgumentParser(description='Evaluación offline de una política')
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, nargs='+')
    p.add_argument('--split', choices=['test', 'val', 'train', 'all'], default='test',
                   help='usa la misma semilla/fracciones que el entrenamiento')
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--test-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None, help='guardar métricas en JSON')
    args = p.parse_args(argv)

    policy = ILPolicy.load(args.model)
    data = load_npz(args.data, policy.cfg)
    if args.split == 'all':
        idx = np.arange(len(data['v']))
    else:
        tr, va, te = split_by_episode(data['episode_id'], args.val_frac, args.test_frac, args.seed)
        idx = {'train': tr, 'val': va, 'test': te}[args.split]

    metrics = compute_metrics(policy, data, idx)
    metrics.update(measure_latency(policy))
    metrics['split'] = args.split
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
