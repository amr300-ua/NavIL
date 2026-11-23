"""Mundo 2D mínimo para probar il_control sin Gazebo (stub del módulo 1).

Habitación rectangular con obstáculos circulares, robot uniciclo y LiDAR por
raycasting. No sustituye a Gazebo: solo permite cerrar el bucle
sensores -> política -> mux -> robot en el ciclo 1.
"""

import math
from typing import Sequence

import numpy as np


class SimWorld:

    def __init__(self, half: float = 4.0, obstacles: Sequence[float] = (),
                 robot_radius: float = 0.18):
        if len(obstacles) % 3:
            raise ValueError('obstacles debe ser una lista plana [x, y, r, x, y, r, ...]')
        self.half = float(half)
        obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
        self.centers, self.radii = obs[:, :2], obs[:, 2]
        self.robot_radius = robot_radius

    def clearance(self, x: float, y: float) -> float:
        """Distancia del centro del robot a la superficie más cercana."""
        d = self.half - max(abs(x), abs(y))
        if len(self.radii):
            d = min(d, float((np.hypot(self.centers[:, 0] - x, self.centers[:, 1] - y)
                              - self.radii).min()))
        return d

    def in_collision(self, x: float, y: float) -> bool:
        return self.clearance(x, y) < self.robot_radius

    def raycast(self, x: float, y: float, heading: float, angles: np.ndarray,
                max_range: float) -> np.ndarray:
        th = heading + angles
        d = np.stack([np.cos(th), np.sin(th)], axis=1)
        p = np.array([x, y])
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(d[:, 0] > 0, (self.half - x) / d[:, 0], (-self.half - x) / d[:, 0])
            ty = np.where(d[:, 1] > 0, (self.half - y) / d[:, 1], (-self.half - y) / d[:, 1])
        tx = np.where(np.isfinite(tx) & (tx > 0), tx, np.inf)
        ty = np.where(np.isfinite(ty) & (ty > 0), ty, np.inf)
        t = np.minimum(tx, ty)
        if len(self.radii):
            oc = p - self.centers
            b = d @ oc.T
            c = (oc ** 2).sum(1) - self.radii ** 2
            disc = b ** 2 - c
            with np.errstate(invalid='ignore'):
                tc = -b - np.sqrt(disc)
            tc = np.where((disc >= 0) & (tc > 0), tc, np.inf)
            t = np.minimum(t, tc.min(axis=1))
        return np.where(t > max_range, np.inf, t)  # sin retorno -> inf (REP 117)


def integrate_unicycle(x: float, y: float, yaw: float, v: float, w: float, dt: float):
    """Integración exacta del uniciclo con (v, w) constantes durante dt."""
    if abs(w) < 1e-6:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    yaw2 = yaw + w * dt
    r = v / w
    x2 = x + r * (math.sin(yaw2) - math.sin(yaw))
    y2 = y - r * (math.cos(yaw2) - math.cos(yaw))
    return x2, y2, math.atan2(math.sin(yaw2), math.cos(yaw2))
