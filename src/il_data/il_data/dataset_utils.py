"""Transformaciones ROS independientes de las dependencias de ROS 2."""

import math
from collections.abc import Sequence

from .schema import NUM_LECTURAS_LIDAR

"""
Imagina que Lidar original devuelve 360 lecturas, 
pero nuestra red neuronal espera 90.

La función remuestrear_lidar transforma las 360 lecturas en 90 valores.

Lo hace mediante interpolación lineal: calcula valores intermedios
para adaptar el número de lecturas.

Necesitamos que tofas las obsevaciones tengan una dimension fija 
para poder introducirlas en la red neuronal.
"""

def remuestrear_lidar(
    rangos: Sequence[float], rango_minimo: float, rango_maximo: float
) -> list[float]:
    """Normaliza y remuestrea un LaserScan a la longitud del esquema."""
    if not rangos:
        raise ValueError("El LaserScan no contiene lecturas")
    if (
        not math.isfinite(rango_minimo)
        or not math.isfinite(rango_maximo)
        or rango_minimo < 0
        or rango_maximo <= rango_minimo
    ):
        raise ValueError("Los límites de rango del LaserScan no son válidos")

    lecturas = []
    for rango in rangos:
        valor = float(rango)
        if math.isnan(valor) or valor == math.inf:
            valor = rango_maximo
        elif valor == -math.inf:
            valor = rango_minimo
        lecturas.append(min(rango_maximo, max(rango_minimo, valor)))

    if len(lecturas) == 1:
        return lecturas * NUM_LECTURAS_LIDAR

    resultado = []
    ultimo_indice = len(lecturas) - 1
    for indice_destino in range(NUM_LECTURAS_LIDAR):
        posicion = indice_destino * ultimo_indice / (NUM_LECTURAS_LIDAR - 1)
        indice_inferior = int(posicion)
        indice_superior = min(indice_inferior + 1, ultimo_indice)
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
