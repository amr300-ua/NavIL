"""Transformaciones ROS independientes de las dependencias de ROS 2."""

import math
from collections.abc import Sequence

import numpy as np

from .schema import NUM_LECTURAS_LIDAR, RANGO_MAXIMO_LIDAR_METROS


def laserscan_a_sectores(
    rangos: Sequence[float],
    angulo_minimo: float,
    incremento_angular: float,
    rango_minimo: float,
    rango_sensor_maximo: float,
    rango_maximo: float = RANGO_MAXIMO_LIDAR_METROS,
    desfase_yaw: float = 0.0,
) -> list[float]:
    """Reduce un LaserScan a 90 sectores como il_control durante inferencia.

    Asigna cada rayo al sector angular más cercano y conserva la distancia
    mínima para no borrar obstáculos finos. Los rayos debajo de range_min y los
    NaN se ignoran; +inf/sin retorno se trata como espacio libre.
    """
    if len(rangos) == 0:
        raise ValueError("El LaserScan no contiene lecturas")
    if (
        not math.isfinite(angulo_minimo)
        or not math.isfinite(incremento_angular)
        or incremento_angular == 0
        or not math.isfinite(rango_minimo)
        or rango_minimo < 0
        or not math.isfinite(rango_sensor_maximo)
        or rango_sensor_maximo <= rango_minimo
        or not math.isfinite(rango_maximo)
        or rango_maximo <= 0
        or not math.isfinite(desfase_yaw)
    ):
        raise ValueError("Los ángulos o el rango máximo del LaserScan no son válidos")

    lecturas = np.asarray(rangos, dtype=np.float64).copy()
    limpias = np.full_like(lecturas, np.nan)
    finitas = np.isfinite(lecturas)
    validas = finitas & (lecturas >= rango_minimo) & (
        lecturas <= rango_sensor_maximo
    )
    limpias[validas] = lecturas[validas]
    limpias[finitas & (lecturas > rango_sensor_maximo)] = rango_maximo
    limpias[np.isposinf(lecturas)] = rango_maximo
    limpias[np.isneginf(lecturas)] = 0.0
    limpias = np.clip(limpias, 0.0, rango_maximo)

    angulos = (
        angulo_minimo
        + np.arange(lecturas.size) * incremento_angular
        + desfase_yaw
    )
    ancho_sector = 2.0 * math.pi / NUM_LECTURAS_LIDAR
    sectores = np.full(NUM_LECTURAS_LIDAR, np.inf)
    indices = (
        np.rint(np.mod(angulos, 2.0 * math.pi) / ancho_sector)
        .astype(np.int64)
        % NUM_LECTURAS_LIDAR
    )
    validas = np.isfinite(limpias)
    np.minimum.at(sectores, indices[validas], limpias[validas])
    sectores[~np.isfinite(sectores)] = rango_maximo
    return sectores.astype(np.float32).tolist()


def remuestrear_lidar(
    rangos: Sequence[float],
    angulo_minimo: float,
    incremento_angular: float,
    rango_maximo: float = RANGO_MAXIMO_LIDAR_METROS,
    desfase_yaw: float = 0.0,
) -> list[float]:
    """Interpola cada 4 grados, como el preprocesado actual de il_control."""
    if len(rangos) == 0:
        raise ValueError("El LaserScan no contiene lecturas")
    angulo_minimo = float(angulo_minimo) + float(desfase_yaw)
    incremento_angular = float(incremento_angular)
    if (
        not math.isfinite(angulo_minimo)
        or not math.isfinite(incremento_angular)
        or incremento_angular == 0
        or not math.isfinite(rango_maximo)
        or rango_maximo <= 0
    ):
        raise ValueError("Los ángulos o el rango máximo del LaserScan no son válidos")

    lecturas = np.asarray(rangos, dtype=np.float64)
    lecturas = np.where(
        np.isnan(lecturas) | np.isposinf(lecturas), rango_maximo, lecturas
    )
    lecturas = np.where(np.isneginf(lecturas), 0.0, lecturas)
    lecturas = np.clip(lecturas, 0.0, rango_maximo)

    cantidad = lecturas.size
    paso = abs(incremento_angular)
    vuelta_completa = paso * cantidad >= 2.0 * math.pi - paso * 1.01
    objetivos = np.arange(NUM_LECTURAS_LIDAR) * (
        2.0 * math.pi / NUM_LECTURAS_LIDAR
    )
    resultado = np.empty(NUM_LECTURAS_LIDAR, dtype=np.float64)
    for indice, angulo in enumerate(objetivos):
        if vuelta_completa:
            posicion = ((angulo - angulo_minimo) / incremento_angular) % cantidad
        else:
            signo = 1.0 if incremento_angular > 0 else -1.0
            posicion = (
                ((angulo - angulo_minimo) * signo) % (2.0 * math.pi)
            ) / paso
            if posicion > cantidad - 1 + 1e-9:
                resultado[indice] = rango_maximo
                continue
            posicion = min(posicion, cantidad - 1)

        inferior = int(math.floor(posicion))
        superior = (
            (inferior + 1) % cantidad
            if vuelta_completa
            else min(inferior + 1, cantidad - 1)
        )
        fraccion = posicion - inferior
        resultado[indice] = (
            lecturas[inferior] * (1.0 - fraccion)
            + lecturas[superior] * fraccion
        )
    return resultado.astype(np.float32).tolist()


def preprocesar_lidar(
    rangos: Sequence[float],
    angulo_minimo: float,
    incremento_angular: float,
    rango_minimo: float,
    rango_sensor_maximo: float,
    rango_maximo: float = RANGO_MAXIMO_LIDAR_METROS,
    desfase_yaw: float = 0.0,
    reduccion: str = "interp",
) -> list[float]:
    """Aplica el modo de reducción compartido con il_control."""
    if reduccion == "interp":
        return remuestrear_lidar(
            rangos,
            angulo_minimo,
            incremento_angular,
            rango_maximo,
            desfase_yaw,
        )
    if reduccion == "min":
        return laserscan_a_sectores(
            rangos,
            angulo_minimo,
            incremento_angular,
            rango_minimo,
            rango_sensor_maximo,
            rango_maximo,
            desfase_yaw,
        )
    raise ValueError(f"Reducción LiDAR no válida: {reduccion}")


""" 
Esta función calcula dos cosas: 
- Distancia desde el robot hasta el objetivo.
- Ángulo del objetivo respecto a la orientacion acutal del robot.

La red neuronal necesita saber hacia donde tiene que dirigirse el robot,
no solamente que obstaculos tiene delante
"""
def calcular_objetivo_relativo(
    posicion_robot: object,
    orientacion_robot: object,
    posicion_objetivo: object,
) -> tuple[float, float]:
    """Devuelve distancia y ángulo del objetivo en el marco del robot."""
    dx = float(posicion_objetivo.x) - float(posicion_robot.x)
    dy = float(posicion_objetivo.y) - float(posicion_robot.y)
    qx = float(orientacion_robot.x)
    qy = float(orientacion_robot.y)
    qz = float(orientacion_robot.z)
    qw = float(orientacion_robot.w)
    if not all(math.isfinite(valor) for valor in (dx, dy, qx, qy, qz, qw)):
        raise ValueError("La pose del robot o el objetivo contiene valores no finitos")

    yaw = yaw_desde_cuaternion(qx, qy, qz, qw)
    distancia = math.hypot(dx, dy)
    angulo = math.atan2(
        math.sin(math.atan2(dy, dx) - yaw),
        math.cos(math.atan2(dy, dx) - yaw),
    )
    return distancia, angulo


def yaw_desde_cuaternion(x: float, y: float, z: float, w: float) -> float:
    """Calcula el yaw de un cuaternión, normalizándolo antes."""
    valores = (float(x), float(y), float(z), float(w))
    if not all(math.isfinite(valor) for valor in valores):
        raise ValueError("La orientación contiene valores no finitos")
    norma = math.sqrt(sum(valor * valor for valor in valores))
    if norma == 0:
        raise ValueError("La orientación no es un cuaternión válido")
    x, y, z, w = (componente / norma for componente in valores)
    return math.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z),
    )
