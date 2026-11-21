"""Genera un dataset determinista para probar el contrato del ciclo 1."""

import argparse
import math
import random

from .dataset_io import guardar_jsonl
from .schema import MuestraNavegacion, NUM_LECTURAS_LIDAR

""" 
Este archivo crea muestras artificial sin necesitar Gazebo.
Sirve para comprobar que el sistema de datos funciona antes de dispones
de grabaciones reales.
"""

def generar_muestra(indice: int, generador: random.Random) -> MuestraNavegacion:
    """Crea una observación ficticia con valores coherentes y reproducibles."""
    angulo_objetivo = generador.uniform(-math.pi, math.pi)
    distancia_objetivo = generador.uniform(0.5, 5.0)
    obstaculo = generador.uniform(0.25, 4.0)
    lecturas_lidar = [
        max(0.12, min(8.0, obstaculo + generador.uniform(-0.2, 0.2)))
        for _ in range(NUM_LECTURAS_LIDAR)
    ]
    velocidad_lineal = min(0.22, distancia_objetivo * 0.15) * max(
        0.0, math.cos(angulo_objetivo)
    )
    velocidad_angular = max(-1.0, min(1.0, 1.5 * angulo_objetivo))
    return MuestraNavegacion(
        marca_temporal=indice * 0.1,
        id_episodio="episodio_ficticio_001",
        lecturas_lidar=lecturas_lidar,
        distancia_objetivo=distancia_objetivo,
        angulo_objetivo=angulo_objetivo,
        velocidad_lineal_experta=velocidad_lineal,
        velocidad_angular_experta=velocidad_angular,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera ejemplos sintéticos para probar el pipeline NavIL."
    )
    parser.add_argument("--output", default="dataset_ficticio.jsonl")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples debe ser mayor que cero")
    generador = random.Random(args.seed)
    muestras = (
        generar_muestra(indice, generador) for indice in range(args.samples)
    )
    cantidad = guardar_jsonl(muestras, args.output)
    print(f"Dataset ficticio guardado: {cantidad} muestras en {args.output}")


if __name__ == "__main__":
    main()
