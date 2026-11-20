"""Generador de datos ficticios (ciclo 1)"""

import argparse
import math
import os

import numpy as np

from il_policy.contract import ContractConfig, scan_angles, wrap_angle

ROBOT_RADIUS = 0.18  # m (TurtleBot 4 ≈ 0.17 m)


class World:
    """Habitación [-half, half]^2 con obstáculos circulares."""

    def __init__(self, rng: np.random.Generator, half: float = 4.0,
                 n_obstacles=(3, 8)):
        self.half = half
        k = rng.integers(n_obstacles[0], n_obstacles[1] + 1)
        self.centers = rng.uniform(-half + 0.8, half - 0.8, size=(k, 2))
        self.radii = rng.uniform(0.2, 0.5, size=k)

    def clearance(self, p: np.ndarray) -> float:
        """Distancia desde p a la superficie más cercana (paredes u obstáculos)."""
        walls = self.half - np.abs(p)
        d = float(walls.min())
        if len(self.radii):
            d = min(d, (np.linalg.norm(self.centers - p, axis=1) - self.radii).min())
        return float(d)

    def raycast(self, p: np.ndarray, heading: float, angles: np.ndarray,
                max_range: float) -> np.ndarray:
        """Distancia a lo largo de cada rayo (vectorizado)."""
        th = heading + angles
        d = np.stack([np.cos(th), np.sin(th)], axis=1)  # (R, 2)

        # Paredes: x = ±half, y = ±half
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(d[:, 0] > 0, (self.half - p[0]) / d[:, 0],
                          (-self.half - p[0]) / d[:, 0])
            ty = np.where(d[:, 1] > 0, (self.half - p[1]) / d[:, 1],
                          (-self.half - p[1]) / d[:, 1])
        tx = np.where(np.isfinite(tx) & (tx > 0), tx, np.inf)
        ty = np.where(np.isfinite(ty) & (ty > 0), ty, np.inf)
        t = np.minimum(tx, ty)

        # Círculos: |p + t d - c|^2 = r^2
        if len(self.radii):
            oc = p - self.centers  # (K, 2)
            b = d @ oc.T  # (R, K)
            c = (oc ** 2).sum(1) - self.radii ** 2  # (K,)
            disc = b ** 2 - c
            with np.errstate(invalid='ignore'):
                tc = -b - np.sqrt(disc)
            tc = np.where((disc >= 0) & (tc > 0), tc, np.inf)
            t = np.minimum(t, tc.min(axis=1))
        return np.minimum(t, max_range)

    def sample_free(self, rng: np.random.Generator, margin: float = 0.45) -> np.ndarray:
        for _ in range(1000):
            p = rng.uniform(-self.half + margin, self.half - margin, size=2)
            if self.clearance(p) > margin:
                return p
        raise RuntimeError('No se encontró posición libre')


def expert_action(scan: np.ndarray, goal: np.ndarray, angles: np.ndarray,
                  cfg: ContractConfig):
    """Controlador experto reactivo. Solo usa (scan, goal): es imitable."""
    dist, alpha = float(goal[0]), float(goal[1])
    d_safe = 0.8

    front_mask = np.abs(angles) < math.pi / 2
    near = front_mask & (scan < d_safe)
    # Lado del obstáculo: +1 izquierda, -1 derecha. Para los rayos casi
    # frontales se decide según qué lado está más libre (evita titubeos).
    left_free = scan[(angles > 0) & front_mask].mean()
    right_free = scan[(angles < 0) & front_mask].mean()
    blocked_side = 1.0 if left_free < right_free else -1.0
    sgn = np.where(np.abs(angles) < math.radians(15), blocked_side,
                   np.where(angles >= 0, 1.0, -1.0))
    # Obstáculo a la izquierda (ángulo > 0) => girar a la derecha (w < 0)
    rep = np.sum(-sgn[near] * (d_safe - scan[near]) / d_safe * np.cos(angles[near]))

    w = 1.5 * alpha + 1.2 * rep

    front = scan[np.abs(angles) < math.radians(30)].min()
    v = cfg.v_max
    v *= np.clip((front - 0.30) / 0.6, 0.0, 1.0)            # frena ante obstáculos
    v *= np.clip(dist / 0.6, 0.25, 1.0)                     # frena al llegar
    v *= np.clip(1.0 - abs(alpha) / (math.pi / 2), 0.0, 1.0)  # gira antes de avanzar

    return float(np.clip(v, -cfg.v_max, cfg.v_max)), float(np.clip(w, -cfg.w_max, cfg.w_max))


def run_episode(rng, cfg: ContractConfig, angles, dt=0.1, max_steps=600,
                noise_std=0.01):
    world = World(rng)
    pos = world.sample_free(rng)
    for _ in range(100):
        goal_xy = world.sample_free(rng)
        if np.linalg.norm(goal_xy - pos) > 1.5:
            break
    heading = rng.uniform(-math.pi, math.pi)

    rows = {'scan': [], 'goal': [], 'v': [], 'w': []}
    outcome = 'timeout'
    for _ in range(max_steps):
        scan = world.raycast(pos, heading, angles, cfg.scan_max_range)
        scan = np.clip(scan + rng.normal(0, noise_std, scan.shape), 0.0, cfg.scan_max_range)
        delta = goal_xy - pos
        dist = float(np.linalg.norm(delta))
        alpha = float(wrap_angle(math.atan2(delta[1], delta[0]) - heading))
        goal = np.array([dist, alpha], dtype=np.float32)

        if dist < 0.25:
            outcome = 'success'
            break

        v, w = expert_action(scan, goal, angles, cfg)
        rows['scan'].append(scan.astype(np.float32))
        rows['goal'].append(goal)
        rows['v'].append(v)
        rows['w'].append(w)

        heading = float(wrap_angle(heading + w * dt))
        pos = pos + v * dt * np.array([math.cos(heading), math.sin(heading)])
        if world.clearance(pos) < ROBOT_RADIUS:
            outcome = 'collision'
            break
    return rows, outcome


def generate_dataset(n_episodes: int = 200, seed: int = 0,
                     cfg: ContractConfig = None, max_steps: int = 600,
                     only_success: bool = True) -> dict:
    """Devuelve un dict con las claves del contrato.

    ``episode_outcome`` guarda el resultado de cada episodio intentado.
    Con ``only_success`` se descartan las demostraciones fallidas (igual que
    habría que hacer con las rutas de Nav2 que no llegan al objetivo).
    """
    cfg = cfg or ContractConfig()
    rng = np.random.default_rng(seed)
    angles = scan_angles(cfg.scan_size)
    out = {'scan': [], 'goal': [], 'v': [], 'w': [], 'episode_id': []}
    outcomes = []
    for ep in range(n_episodes):
        rows, outcome = run_episode(rng, cfg, angles, max_steps=max_steps)
        outcomes.append(outcome)
        n = len(rows['v'])
        if n == 0 or (only_success and outcome != 'success'):
            continue
        for k in ('scan', 'goal', 'v', 'w'):
            out[k].extend(rows[k])
        out['episode_id'].extend([ep] * n)
    data = {
        'scan': np.asarray(out['scan'], dtype=np.float32),
        'goal': np.asarray(out['goal'], dtype=np.float32),
        'v': np.asarray(out['v'], dtype=np.float32),
        'w': np.asarray(out['w'], dtype=np.float32),
        'episode_id': np.asarray(out['episode_id'], dtype=np.int32),
        'episode_outcome': np.asarray(outcomes),
    }
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description='Genera un dataset ficticio con el formato del contrato.')
    p.add_argument('--out', default='data/dummy.npz')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--keep-failures', action='store_true',
                   help='guardar también episodios con colisión/timeout')
    args = p.parse_args(argv)

    data = generate_dataset(args.episodes, args.seed, max_steps=args.max_steps,
                            only_success=not args.keep_failures)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)

    oc = data['episode_outcome']
    n_ep = len(np.unique(data['episode_id']))
    print(f'Guardado {args.out}: {len(data["v"])} muestras de {n_ep}/{args.episodes} episodios '
          f'(éxito {np.mean(oc == "success"):.0%}, colisión {np.mean(oc == "collision"):.0%}, '
          f'timeout {np.mean(oc == "timeout"):.0%})')


if __name__ == '__main__':
    main()
