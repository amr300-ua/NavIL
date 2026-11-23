"""Construcción de la observación del modelo a partir de los sensores.

Funciones puras (sin ROS) para poder probarlas con pytest y, si el equipo lo
acuerda, compartirlas con ``il_data`` para que entrenamiento e inferencia usen
exactamente el mismo preprocesado (evita el *train/serve skew*).

Convención (idéntica a ``il_policy/contract.py``):

* ``scan[i]`` es la distancia mínima en el sector centrado en ``i * 360/N``
  grados, medidos en sentido antihorario desde el frente del robot
  (``scan[0]`` delante, ``scan[N/4]`` a la izquierda, ``scan[N/2]`` detrás).
* Los ángulos se expresan en el marco del robot (``base_link``), no en el del
  LiDAR: en el TurtleBot 4 el ``rplidar_link`` puede ir girado respecto a la
  base, así que se aplica ``yaw_offset`` (obtenido de TF en el nodo).
* ``goal = [distancia (m), ángulo (rad)]``, ángulo positivo a la izquierda,
  en ``[-pi, pi)``.
"""

import math
from typing import Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(a):
    """Normaliza ángulos a [-pi, pi). Acepta escalares o arrays."""
    return (np.asarray(a) + math.pi) % TWO_PI - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rad) de un cuaternión. Misma fórmula que ``il_data``."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError('Cuaternión no válido')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clean_ranges(ranges: Sequence[float], range_min: float, range_max: float,
                 max_range: float) -> np.ndarray:
    """Limpia las lecturas siguiendo REP 117.

    * ``+inf`` (sin retorno) y lecturas > ``range_max`` -> ``max_range`` (libre).
    * ``-inf`` (objeto demasiado cerca) -> 0.0.
    * ``NaN`` y lecturas finitas < ``range_min`` (ruido / cuerpo del robot) ->
      ``NaN``: se ignoran al agregar por sectores.
    """
    r = np.asarray(ranges, dtype=np.float64).copy()
    out = np.full_like(r, np.nan)
    finite = np.isfinite(r)
    valid = finite & (r >= range_min) & (r <= range_max)
    out[valid] = r[valid]
    out[finite & (r > range_max)] = max_range
    out[np.isposinf(r)] = max_range
    out[np.isneginf(r)] = 0.0
    return np.clip(out, 0.0, max_range)


def laserscan_to_sectors(ranges: Sequence[float], angle_min: float,
                         angle_increment: float, range_min: float,
                         range_max: float, n_sectors: int = 90,
                         max_range: float = 12.0,
                         yaw_offset: float = 0.0) -> np.ndarray:
    """Reduce un ``LaserScan`` a ``n_sectors`` distancias en el marco del robot.

    Cada rayo se asigna al sector más cercano por **ángulo** (no por índice),
    así que funciona igual con 360, 640 o 1440 rayos y con FOV parciales.
    Se toma el **mínimo** de cada sector: es la opción conservadora, porque
    interpolar o promediar puede "borrar" obstáculos finos (patas de mesa).
    Un sector sin lecturas válidas se considera libre (``max_range``), igual
    que ``inf`` en el contrato de ``il_policy``.
    """
    if n_sectors <= 0:
        raise ValueError('n_sectors debe ser positivo')
    r = clean_ranges(ranges, range_min, range_max, max_range)
    if r.size == 0:
        raise ValueError('El LaserScan no contiene lecturas')

    angles = angle_min + np.arange(r.size) * angle_increment + yaw_offset
    width = TWO_PI / n_sectors
    idx = np.rint(np.mod(angles, TWO_PI) / width).astype(np.int64) % n_sectors

    sectors = np.full(n_sectors, np.inf)
    ok = np.isfinite(r)
    np.minimum.at(sectors, idx[ok], r[ok])
    sectors[~np.isfinite(sectors)] = max_range
    return sectors.astype(np.float32)


def scan_to_points(ranges: Sequence[float], angle_min: float,
                   angle_increment: float, range_min: float, range_max: float,
                   yaw_offset: float = 0.0, x_offset: float = 0.0,
                   y_offset: float = 0.0) -> np.ndarray:
    """Convierte un ``LaserScan`` en puntos (N, 2) en el marco del robot.

    Solo incluye retornos reales (finitos y dentro de rango, o ``-inf`` como
    punto pegado al sensor). Lo usa la capa de seguridad, que necesita
    geometría y no el vector de 90 valores.
    """
    r = np.asarray(ranges, dtype=np.float64)
    angles = angle_min + np.arange(r.size) * angle_increment + yaw_offset
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    rr = np.where(np.isneginf(r), 0.0, r)
    keep = valid | np.isneginf(r)
    x = rr[keep] * np.cos(angles[keep]) + x_offset
    y = rr[keep] * np.sin(angles[keep]) + y_offset
    return np.stack([x, y], axis=1)


def transform_point_2d(tx: float, ty: float, yaw: float,
                       px: float, py: float) -> Tuple[float, float]:
    """Aplica la transformada 2D (tx, ty, yaw) al punto (px, py)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * px - s * py, ty + s * px + c * py


def relative_goal(robot_x: float, robot_y: float, robot_yaw: float,
                  goal_x: float, goal_y: float) -> Tuple[float, float]:
    """[distancia, ángulo] del objetivo en el marco del robot.

    Equivalente a ``il_data.dataset_utils.calcular_objetivo_relativo``.
    """
    dx, dy = goal_x - robot_x, goal_y - robot_y
    if not all(math.isfinite(v) for v in (dx, dy, robot_yaw)):
        raise ValueError('Pose del robot u objetivo no finitos')
    return math.hypot(dx, dy), float(wrap_angle(math.atan2(dy, dx) - robot_yaw))


def goal_in_base_frame(gx_base: float, gy_base: float) -> Tuple[float, float]:
    """[distancia, ángulo] de un objetivo ya expresado en ``base_link``."""
    return math.hypot(gx_base, gy_base), float(wrap_angle(math.atan2(gy_base, gx_base)))
