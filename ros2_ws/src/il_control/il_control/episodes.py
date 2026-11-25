"""Lógica (sin ROS) de los episodios de evaluación en bucle cerrado.

* Carga de escenarios con el **mismo formato YAML que navil_expert**
  (``escenarios: <nombre>: waypoints: [{x, y, yaw(grados)}]``), así Nav2 y
  la política se evalúan exactamente sobre las mismas rutas.
* ``EpisodeTracker`` acumula las métricas de un episodio y decide cuándo
  termina: éxito, timeout, atasco o colisión.
* Resumen agregado y escritura a CSV/JSON para la memoria.
"""

import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Waypoint = Tuple[float, float, float]  # x (m), y (m), yaw (grados), marco map

SUCCESS = 'success'
TIMEOUT = 'timeout'
STUCK = 'stuck'
COLLISION = 'collision'
ABORTED = 'aborted'          # Nav2 aborta/rechaza (solo modo experto)
OUTCOMES = (SUCCESS, TIMEOUT, STUCK, COLLISION, ABORTED)


def load_scenarios(path: str) -> Dict[str, List[Waypoint]]:
    """Lee un scenarios.yaml con el formato de navil_expert."""
    import yaml  # PyYAML viene con ROS 2
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    out = {}
    for name, sc in (data.get('escenarios') or {}).items():
        wps = [(float(p['x']), float(p['y']), float(p.get('yaw', 0.0)))
               for p in (sc or {}).get('waypoints', [])]
        if not wps:
            raise ValueError(f"El escenario '{name}' no tiene waypoints")
        out[name] = wps
    if not out:
        raise ValueError(f'{path} no contiene escenarios')
    return out


def waypoints_from_flat(values: Sequence[float]) -> List[Waypoint]:
    """[x1, y1, yaw1, x2, ...] -> lista de waypoints (parámetro ROS plano)."""
    if len(values) % 3:
        raise ValueError('waypoints debe tener 3 valores por objetivo (x, y, yaw_deg)')
    v = [float(x) for x in values]
    return [tuple(v[i:i + 3]) for i in range(0, len(v), 3)]


@dataclass
class EpisodeLimits:
    goal_tolerance: float = 0.25      # m (igual que il_policy_node)
    timeout_base: float = 30.0        # s
    timeout_per_meter: float = 10.0   # s por metro de distancia en línea recta
    stuck_window: float = 15.0        # s
    stuck_distance: float = 0.10      # m que hay que avanzar en stuck_window


@dataclass
class EpisodeResult:
    episode_id: str
    run_id: str
    scenario: str
    controller: str
    policy: str
    repetition: int
    waypoint_index: int
    waypoint_total: int
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    straight_dist_m: float
    outcome: str
    success: bool
    time_s: float
    path_length_m: float
    path_efficiency: float          # avance neto hacia el objetivo / recorrido (1 = ideal)
    final_dist_m: float
    collisions: int
    estop_events: int
    min_obstacle_dist_m: float
    mean_v: float
    inference_ms_mean: float
    inference_ms_p95: float
    inference_hz: float             # 1000 / media; capacidad, no frecuencia real

    def to_row(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4) if math.isfinite(v) else ''
        return d


@dataclass
class EpisodeTracker:
    """Estado de un episodio en curso. Tiempos en segundos (reloj ROS)."""

    goal: Tuple[float, float]
    start_time: float
    start_pose: Tuple[float, float]
    limits: EpisodeLimits = field(default_factory=EpisodeLimits)

    def __post_init__(self):
        self.straight = math.hypot(self.goal[0] - self.start_pose[0],
                                   self.goal[1] - self.start_pose[1])
        self.timeout = self.limits.timeout_base + self.limits.timeout_per_meter * self.straight
        self.path_length = 0.0
        self._last_odom: Optional[Tuple[float, float]] = None
        self._progress: List[Tuple[float, float, float]] = []  # (t, x, y) en map
        self.last_dist = self.straight
        self.last_time = self.start_time
        self.collisions = 0
        self.min_obstacle = math.inf
        self.latencies: List[float] = []
        self.estop_events = 0

    # --------------------------------------------------------------- entradas
    def update_odom(self, x: float, y: float) -> None:
        """Longitud de trayectoria con la odometría (marco odom).

        Se usa odom y no map porque las correcciones de AMCL producen saltos
        en map que inflarían la longitud recorrida.
        """
        if self._last_odom is not None:
            step = math.hypot(x - self._last_odom[0], y - self._last_odom[1])
            if step < 1.0:  # descarta saltos espurios (reinicio de odometría)
                self.path_length += step
        self._last_odom = (x, y)

    def update_map(self, t: float, x: float, y: float) -> None:
        self.last_time = t
        self.last_dist = math.hypot(self.goal[0] - x, self.goal[1] - y)
        self._progress.append((t, x, y))
        while self._progress and t - self._progress[0][0] > self.limits.stuck_window:
            self._progress.pop(0)

    def add_collision(self) -> None:
        self.collisions += 1

    def add_scan_min(self, d: float) -> None:
        if math.isfinite(d):
            self.min_obstacle = min(self.min_obstacle, d)

    def add_latency(self, ms: float) -> None:
        if math.isfinite(ms):
            self.latencies.append(ms)

    # --------------------------------------------------------------- decisión
    def check(self, t: float, stop_on_collision: bool = True,
              use_distance_success: bool = True) -> Optional[str]:
        """Devuelve el motivo de fin del episodio o None si continúa."""
        if use_distance_success and self.last_dist < self.limits.goal_tolerance:
            return SUCCESS
        if stop_on_collision and self.collisions > 0:
            return COLLISION
        elapsed = t - self.start_time
        if elapsed > self.timeout:
            return TIMEOUT
        if elapsed > self.limits.stuck_window and len(self._progress) >= 2:
            t0, x0, y0 = self._progress[0]
            _, x1, y1 = self._progress[-1]
            if math.hypot(x1 - x0, y1 - y0) < self.limits.stuck_distance:
                return STUCK
        return None

    def result(self, outcome: str, **meta) -> EpisodeResult:
        elapsed = self.last_time - self.start_time
        lat = np.asarray(self.latencies, dtype=float)
        mean_lat = float(lat.mean()) if lat.size else math.nan
        return EpisodeResult(
            outcome=outcome, success=outcome == SUCCESS, time_s=elapsed,
            start_x=self.start_pose[0], start_y=self.start_pose[1],
            goal_x=self.goal[0], goal_y=self.goal[1],
            straight_dist_m=self.straight, path_length_m=self.path_length,
            # recortado a 1: el primer paso antes de la primera lectura de odom
            # no se integra y puede dar 1.00001
            path_efficiency=(min(1.0, max(0.0, self.straight - self.last_dist)
                                 / self.path_length)
                             if self.path_length > 1e-6 else math.nan),
            final_dist_m=self.last_dist, collisions=self.collisions,
            estop_events=self.estop_events, min_obstacle_dist_m=self.min_obstacle,
            mean_v=self.path_length / elapsed if elapsed > 0 else 0.0,
            inference_ms_mean=mean_lat,
            inference_ms_p95=float(np.percentile(lat, 95)) if lat.size else math.nan,
            inference_hz=1000.0 / mean_lat if lat.size and mean_lat > 0 else math.nan,
            **meta)


def summarize(results: Sequence[EpisodeResult]) -> dict:
    """Métricas agregadas. Tiempo y longitud solo sobre episodios con éxito
    (en los fallidos no son comparables)."""
    n = len(results)
    ok = [r for r in results if r.success]

    def mean(xs):
        xs = [x for x in xs if math.isfinite(x)]
        return round(float(np.mean(xs)), 4) if xs else None

    def std(xs):
        xs = [x for x in xs if math.isfinite(x)]
        return round(float(np.std(xs)), 4) if len(xs) > 1 else None

    return {
        'episodes': n,
        'successes': len(ok),
        'success_rate': round(len(ok) / n, 4) if n else None,
        'outcomes': {o: sum(r.outcome == o for r in results) for o in OUTCOMES},
        'collisions_total': sum(r.collisions for r in results),
        'episodes_with_collision': sum(r.collisions > 0 for r in results),
        'estop_events_total': sum(r.estop_events for r in results),
        'time_s_mean_success': mean([r.time_s for r in ok]),
        'time_s_std_success': std([r.time_s for r in ok]),
        'path_length_m_mean_success': mean([r.path_length_m for r in ok]),
        'path_efficiency_mean_success': mean([r.path_efficiency for r in ok]),
        'min_obstacle_dist_m_min': (round(min(r.min_obstacle_dist_m for r in results), 4)
                                    if results and any(math.isfinite(r.min_obstacle_dist_m)
                                                       for r in results) else None),
        'inference_ms_mean': mean([r.inference_ms_mean for r in results]),
        'inference_ms_p95_max': (max((r.inference_ms_p95 for r in results
                                      if math.isfinite(r.inference_ms_p95)), default=None)),
    }


class ResultsWriter:
    """episodes.csv (una fila por episodio, se escribe al terminar cada uno,
    así no se pierde nada si se corta la simulación) + summary.json."""

    def __init__(self, out_dir: str, meta: dict):
        self.dir = Path(out_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / 'episodes.csv'
        self.meta = meta
        self.results: List[EpisodeResult] = []

    def add(self, r: EpisodeResult) -> None:
        self.results.append(r)
        row = r.to_row()
        new = not self.csv_path.exists()
        with self.csv_path.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def write_summary(self) -> dict:
        s = {'meta': self.meta, 'summary': summarize(self.results)}
        (self.dir / 'summary.json').write_text(json.dumps(s, indent=2, ensure_ascii=False),
                                               encoding='utf-8')
        return s
