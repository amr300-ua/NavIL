"""Backends de política detrás de una interfaz común ``predict(obs) -> (v, w)``.

El nodo de inferencia no sabe qué hay detrás: un mock para el ciclo 1 o el
``ILPolicy`` de ``il_policy`` (rama feature/il-policy) a partir del ciclo 2.
Así se cambia de modelo con un parámetro, sin tocar el nodo.
"""

import math
from typing import Mapping, Tuple

import numpy as np

POLICY_TYPES = ('mock_reactive', 'mock_goal', 'zero', 'il_policy')


class PolicyBackend:
    """Interfaz mínima. ``scan_size`` y ``max_range`` definen el preprocesado."""

    name = 'base'
    scan_size = 90
    max_range = 12.0
    v_max = 0.30
    w_max = 1.50

    def predict(self, observation: Mapping) -> Tuple[float, float]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


def _sector_angles(n: int) -> np.ndarray:
    a = np.arange(n) * (2.0 * math.pi / n)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ZeroPolicy(PolicyBackend):
    """Siempre (0, 0). Útil para probar el multiplexor y los watchdogs."""

    name = 'zero'

    def predict(self, observation: Mapping) -> Tuple[float, float]:
        return 0.0, 0.0


class MockGoalPolicy(PolicyBackend):
    """Controlador proporcional hacia el objetivo, **sin** evitar obstáculos.

    Sirve para comprobar que la capa de seguridad frena al robot.
    """

    name = 'mock_goal'

    def __init__(self, k_w: float = 1.5, v_max: float = 0.25, w_max: float = 1.2):
        self.k_w, self.v_max, self.w_max = k_w, v_max, w_max

    def predict(self, observation: Mapping) -> Tuple[float, float]:
        dist, ang = (float(x) for x in observation['goal'])
        w = float(np.clip(self.k_w * ang, -self.w_max, self.w_max))
        v = self.v_max * max(0.0, math.cos(ang)) * min(1.0, dist / 0.5)
        return v, w


class MockReactivePolicy(MockGoalPolicy):
    """Ir al objetivo esquivando obstáculos (*follow-the-gap* simplificado).

    1. Una dirección es "transitable" si el pasillo del ancho del robot
       (``2 * half_width``) está libre hasta ``lookahead`` metros (o hasta el
       objetivo, si está más cerca).
    2. Se elige el sector transitable más próximo a la dirección del objetivo
       y se gira hacia él; la velocidad lineal baja con el ángulo de giro y
       con la distancia libre delante.

    No pretende ser un buen controlador: es un sustituto determinista del
    modelo para probar el nodo en bucle cerrado sin red entrenada.
    """

    name = 'mock_reactive'

    def __init__(self, lookahead: float = 1.0, half_width: float = 0.25, **kw):
        super().__init__(**kw)
        self.lookahead, self.half_width = lookahead, half_width

    def predict(self, observation: Mapping) -> Tuple[float, float]:
        scan = np.asarray(observation['scan'], dtype=np.float64)
        dist, goal_ang = (float(x) for x in observation['goal'])
        n = scan.size
        ang = _sector_angles(n)
        reach = min(self.lookahead, dist)
        # Puntos de obstáculo (marco robot) proyectados sobre cada dirección
        # candidata: bloqueada si alguno cae en el pasillo [0, reach] x ±half_width.
        px, py = scan * np.cos(ang), scan * np.sin(ang)
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        along = c * px[None, :] + s * py[None, :]
        lateral = -s * px[None, :] + c * py[None, :]
        inside = (along > 0) & (np.abs(lateral) < self.half_width)
        free_len = np.where(inside, along, np.inf).min(axis=1)  # libre en cada dirección
        passable = free_len > reach
        if np.any(passable):
            err = np.abs((ang - goal_ang + math.pi) % (2 * math.pi) - math.pi)
            i = int(np.flatnonzero(passable)[np.argmin(err[passable])])
            target, free = float(ang[i]), float(free_len[i])
        else:  # encerrado: girar hacia el lado más libre sin avanzar
            target = math.copysign(math.pi / 2, float(np.argmax(scan) < n / 2) - 0.5)
            free = 0.0
        w = float(np.clip(self.k_w * target, -self.w_max, self.w_max))
        v = (self.v_max * max(0.0, math.cos(target)) ** 2 * min(1.0, dist / 0.5)
             * float(np.clip((free - 0.3) / 0.7, 0.0, 1.0)))
        return v, w


class ILPolicyBackend(PolicyBackend):
    """Adaptador del ``ILPolicy`` de ``il_policy`` (módulo 3).

    El import es perezoso: ``il_control`` no depende de PyTorch ni de
    ``il_policy`` salvo que se pida este backend. El preprocesado (nº de
    sectores, rango máximo) se toma del contrato guardado en el checkpoint.
    """

    name = 'il_policy'

    def __init__(self, model_path: str = '', device: str = 'cpu'):
        try:
            from il_policy.policy import ILPolicy
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "policy_type='il_policy' requiere el paquete il_policy y PyTorch "
                f'en el workspace ({e})') from e
        self._policy = (ILPolicy.load(model_path, device=device) if model_path
                        else ILPolicy.untrained())
        cfg = self._policy.cfg
        self.scan_size = int(cfg.scan_size)
        self.max_range = float(cfg.scan_max_range)
        self.v_max, self.w_max = float(cfg.v_max), float(cfg.w_max)
        self.model_path = model_path or '<sin entrenar>'

    def predict(self, observation: Mapping) -> Tuple[float, float]:
        return self._policy.predict(observation)

    def describe(self) -> str:
        return f'il_policy ({self.model_path})'


def make_policy(policy_type: str, model_path: str = '', device: str = 'cpu') -> PolicyBackend:
    if policy_type == 'zero':
        return ZeroPolicy()
    if policy_type == 'mock_goal':
        return MockGoalPolicy()
    if policy_type == 'mock_reactive':
        return MockReactivePolicy()
    if policy_type == 'il_policy':
        return ILPolicyBackend(model_path, device)
    raise ValueError(f"policy_type '{policy_type}' no válido. Opciones: {POLICY_TYPES}")
