"""Transformaciones ROS independientes de las dependencias de ROS 2."""

import math
from collections.abc import Sequence

from .schema import NUM_LECTURAS_LIDAR, RANGO_MAXIMO_LIDAR_METROS


def remuestrear_lidar(
    rangos: Sequence[float],
    angulo_minimo: float,
    incremento_angular: float,
    rango_maximo: float = RANGO_MAXIMO_LIDAR_METROS,
) -> list[float]:
    """Remuestrea LaserScan a rayos cada 4 grados desde el frente, antihorario.

    El resultado cumple el contrato del modelo: scan[0] apunta al frente,
    scan[22] apunta aproximadamente a la izquierda y las distancias están en
    metros. En sectores no cubiertos por un LiDAR de campo de visión parcial,
    se usa el máximo del contrato.
    """
    if not rangos:
        raise ValueError("El LaserScan no contiene lecturas")
    if (
        not math.isfinite(angulo_minimo)
        or not math.isfinite(incremento_angular)
        or incremento_angular == 0
        or not math.isfinite(rango_maximo)
        or rango_maximo <= 0
    ):
        raise ValueError("Los ángulos o el rango máximo del LaserScan no son válidos")

    lecturas = []
    for rango in rangos:
        valor = float(rango)
        if math.isnan(valor) or valor == math.inf:
            valor = rango_maximo
        elif valor == -math.inf:
            valor = 0.0
        lecturas.append(min(rango_maximo, max(0.0, valor)))

    resultado = []
    cantidad = len(lecturas)
    paso = abs(incremento_angular)
    cubre_una_vuelta = paso * cantidad >= (2 * math.pi) - paso * 1.01
    for indice_destino in range(NUM_LECTURAS_LIDAR):
        angulo_destino = indice_destino * 2 * math.pi / NUM_LECTURAS_LIDAR
        if cubre_una_vuelta:
            posicion = (
                (angulo_destino - angulo_minimo) / incremento_angular
            ) % cantidad
        else:
            sentido = 1.0 if incremento_angular > 0 else -1.0
            desplazamiento = (
                (angulo_destino - angulo_minimo) * sentido
            ) % (2 * math.pi)
            posicion = desplazamiento / paso
            if posicion > cantidad - 1 + 1e-9:
                resultado.append(rango_maximo)
                continue
            posicion = min(posicion, cantidad - 1)

        indice_inferior = int(math.floor(posicion))
        indice_superior = indice_inferior + 1
        if cubre_una_vuelta:
            indice_superior %= cantidad
        else:
            indice_superior = min(indice_superior, cantidad - 1)
        fraccion = posicion - indice_inferior
        resultado.append(
            lecturas[indice_inferior] * (1 - fraccion)
            + lecturas[indice_superior] * fraccion
        )
    return resultado


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

    norma = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norma == 0:
        raise ValueError("La orientación del robot no es un cuaternión válido")
    qx, qy, qz, qw = (componente / norma for componente in (qx, qy, qz, qw))
    yaw = math.atan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy * qy + qz * qz),
    )
    distancia = math.hypot(dx, dy)
    angulo = math.atan2(
        math.sin(math.atan2(dy, dx) - yaw),
        math.cos(math.atan2(dy, dx) - yaw),
    )
    return distancia, angulo
