"""Capa de seguridad entre el selector de control y el robot.

Lógica pura (sin ROS): el nodo ``cmd_vel_mux`` le pasa el comando elegido, las
marcas de tiempo de las últimas entradas y los puntos del LiDAR, y recibe el
comando que realmente se envía junto con el motivo.

Orden de las comprobaciones (de más a menos grave):

1. Comando no finito (NaN/inf)            -> parada inmediata.
2. Watchdog del LiDAR (sin /scan reciente) -> parada inmediata. Sin sensor no se
   puede garantizar la parada por obstáculo, así que no se mueve.
3. Timeout del comando de la fuente activa -> parada inmediata. Si el modelo o
   Nav2 se cuelgan, el robot no sigue con la última velocidad.
4. Límites de velocidad (saturación).
5. Proximidad: se comprueba un **pasillo** del ancho del robot en la dirección
   de avance. Zona de frenado progresivo y zona de parada. Solo se limita la
   componente lineal: girar sobre sí mismo sigue permitido (robot circular),
   y así la política puede salir de la situación.
6. Límite de aceleración (rampa), salvo en paradas de seguridad, que son
   inmediatas.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

# Motivos (se publican en /il_control/status y se cuentan en la evaluación)
OK = 'ok'
CLAMPED = 'clamped'
SLOWDOWN = 'slowdown'
ESTOP_FRONT = 'estop_front'
ESTOP_REAR = 'estop_rear'
NO_SCAN = 'scan_timeout'
NO_CMD = 'cmd_timeout'
INVALID_CMD = 'invalid_cmd'
MODE_STOP = 'mode_stop'

SAFETY_STOPS = {ESTOP_FRONT, ESTOP_REAR, NO_SCAN, NO_CMD, INVALID_CMD, MODE_STOP}


@dataclass
class SafetyConfig:
    v_max: float = 0.30           # m/s (igual que il_policy.ContractConfig)
    v_min: float = -0.10          # m/s (marcha atrás limitada)
    w_max: float = 1.50           # rad/s
    max_lin_accel: float = 1.0    # m/s^2   (<= 0 desactiva la rampa)
    max_ang_accel: float = 4.0    # rad/s^2 (<= 0 desactiva la rampa)
    robot_radius: float = 0.18    # m (TurtleBot 4 ~ 0.17 m)
    lateral_margin: float = 0.05  # m extra a cada lado del pasillo
    stop_distance: float = 0.20   # m libres delante del casco -> parada
    slow_distance: float = 0.50   # m libres delante del casco -> empieza a frenar
    scan_timeout: float = 0.5     # s
    cmd_timeout: float = 0.5      # s
    proximity_check: bool = True

    def validate(self) -> None:
        if self.v_max < 0 or self.v_min > 0 or self.w_max <= 0:
            raise ValueError('Límites de velocidad incoherentes')
        if not 0.0 <= self.stop_distance < self.slow_distance:
            raise ValueError('Se requiere 0 <= stop_distance < slow_distance')
        if self.scan_timeout <= 0 or self.cmd_timeout <= 0:
            raise ValueError('Los timeouts deben ser positivos')


@dataclass
class SafetyResult:
    v: float
    w: float
    reason: str
    clearance: float  # m libres en la dirección de avance (inf si no aplica)

    @property
    def is_stop(self) -> bool:
        return self.reason in SAFETY_STOPS


def corridor_clearance(points: Optional[np.ndarray], half_width: float,
                       robot_radius: float, forward: bool = True) -> float:
    """Distancia libre entre el casco del robot y el obstáculo más cercano
    dentro del pasillo de ancho ``2 * half_width`` hacia delante (o detrás).

    ``points`` en el marco del robot, forma (N, 2). Devuelve ``inf`` si no hay
    obstáculos en el pasillo.
    """
    if points is None or len(points) == 0:
        return math.inf
    x = points[:, 0] if forward else -points[:, 0]
    in_corridor = (x > 0.0) & (np.abs(points[:, 1]) <= half_width)
    if not np.any(in_corridor):
        return math.inf
    return float(max(0.0, x[in_corridor].min() - robot_radius))


class SafetyFilter:
    """Filtro con estado (necesita la última salida para la rampa)."""

    def __init__(self, cfg: SafetyConfig):
        cfg.validate()
        self.cfg = cfg
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self._last_v = self._last_w = 0.0
        self._last_t = None

    def _stop(self, now: float, reason: str, clearance: float = math.inf) -> SafetyResult:
        self._last_v, self._last_w, self._last_t = 0.0, 0.0, now
        return SafetyResult(0.0, 0.0, reason, clearance)

    def stop(self, now: float, reason: str = MODE_STOP) -> SafetyResult:
        return self._stop(now, reason)

    def filter(self, v: float, w: float, now: float,
               cmd_stamp: Optional[float], scan_stamp: Optional[float],
               points: Optional[np.ndarray],
               proximity_check: Optional[bool] = None) -> SafetyResult:
        """Devuelve el comando seguro. Tiempos en segundos (mismo reloj)."""
        c = self.cfg
        if not (math.isfinite(v) and math.isfinite(w)):
            return self._stop(now, INVALID_CMD)
        if scan_stamp is None or now - scan_stamp > c.scan_timeout:
            return self._stop(now, NO_SCAN)
        if cmd_stamp is None or now - cmd_stamp > c.cmd_timeout:
            return self._stop(now, NO_CMD)

        reason = OK
        v_c = min(max(v, c.v_min), c.v_max)
        w_c = min(max(w, -c.w_max), c.w_max)
        if v_c != v or w_c != w:
            reason = CLAMPED

        clearance = math.inf
        check = c.proximity_check if proximity_check is None else proximity_check
        if check and v_c != 0.0:
            forward = v_c > 0.0
            clearance = corridor_clearance(points, c.robot_radius + c.lateral_margin,
                                           c.robot_radius, forward)
            if clearance <= c.stop_distance:
                # Parada inmediata de la componente lineal; el giro se mantiene.
                self._last_v, self._last_w, self._last_t = 0.0, w_c, now
                return SafetyResult(0.0, w_c, ESTOP_FRONT if forward else ESTOP_REAR,
                                    clearance)
            if clearance < c.slow_distance:
                k = (clearance - c.stop_distance) / (c.slow_distance - c.stop_distance)
                v_c *= k
                reason = SLOWDOWN

        v_c, w_c = self._ramp(v_c, w_c, now)
        return SafetyResult(v_c, w_c, reason, clearance)

    def _ramp(self, v: float, w: float, now: float):
        c = self.cfg
        if self._last_t is not None:
            dt = max(0.0, now - self._last_t)
            if c.max_lin_accel > 0:
                dv = c.max_lin_accel * dt
                v = min(max(v, self._last_v - dv), self._last_v + dv)
            if c.max_ang_accel > 0:
                dw = c.max_ang_accel * dt
                w = min(max(w, self._last_w - dw), self._last_w + dw)
        self._last_v, self._last_w, self._last_t = v, w, now
        return v, w
