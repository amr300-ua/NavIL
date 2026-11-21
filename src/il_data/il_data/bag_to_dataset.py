"""Convierte un rosbag2 de NavIL a un dataset JSONL validado."""

import argparse
import math
from pathlib import Path
from typing import Any

from .dataset_io import guardar_jsonl
from .dataset_utils import calcular_objetivo_relativo, remuestrear_lidar
from .schema import MuestraNavegacion

"""
Este archivo conecta la grabación con el entrenamiento. 
Hasta ahora hemos grabajo ROS2  pero necesitamos recibir muestras esctructuradas.

Prcedimiento:
1.Abre el ROS Bag.
2.Comprueba que contiene los tópicos necesarios.
3.Lee los mensajes.
4.Mantiene guardadas las últimas observaciones de LiDAR, posición y objetivo.
5.Cuando recibe un comando de velocidad, busca las observaciones anteriores más recientes.
6.Comprueba que el LiDAR y la posición no tengan más de 0,5 segundos de antigüedad.
7.Calcula la distancia y el ángulo al objetivo.
8.Construye una MuestraNavegacion.
9.Guarda todas las muestras en JSONL.

"""

def convertir(
    ruta_bag: str | Path,
    ruta_salida: str | Path,
    scan_topic: str = "/scan",
    cmd_vel_topic: str = "/cmd_vel",
    pose_topic: str = "/amcl_pose",
    goal_topic: str = "/goal_pose",
    antiguedad_maxima: float = 0.5,
) -> int:
    """Convierte las velocidades usando las observaciones más recientes."""
    if not math.isfinite(antiguedad_maxima) or antiguedad_maxima <= 0:
        raise ValueError("La antigüedad máxima debe ser positiva y finita")

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError(
            "La conversión de rosbag requiere ROS 2, rosbag2_py y rclpy."
        ) from error

    lector = rosbag2_py.SequentialReader()
    lector.open(
        rosbag2_py.StorageOptions(uri=str(ruta_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topicos = {entrada.name: entrada.type for entrada in lector.get_all_topics_and_types()}
    requeridos = {
        scan_topic: "sensor_msgs/msg/LaserScan",
        cmd_vel_topic: "geometry_msgs/msg/Twist",
        pose_topic: "geometry_msgs/msg/PoseWithCovarianceStamped",
        goal_topic: "geometry_msgs/msg/PoseStamped",
    }
    ausentes = sorted(set(requeridos) - topicos.keys())
    if ausentes:
        raise ValueError(f"El rosbag no contiene los tópicos necesarios: {ausentes}")
    tipos_incorrectos = {
        nombre: (topicos[nombre], tipo)
        for nombre, tipo in requeridos.items()
        if topicos[nombre] != tipo
    }
    if tipos_incorrectos:
        raise ValueError(f"Tipos de mensaje inesperados: {tipos_incorrectos}")

    tipos = {nombre: get_message(tipo) for nombre, tipo in requeridos.items()}
    ultimo: dict[str, tuple[int, Any]] = {}
    muestras = []
    primera_marca_ns: int | None = None
    id_episodio = Path(ruta_bag).name
    descartadas = 0

    while lector.has_next():
        nombre, datos_serializados, marca_ns = lector.read_next()
        if nombre not in tipos:
            continue
        if primera_marca_ns is None:
            primera_marca_ns = marca_ns
        mensaje = deserialize_message(datos_serializados, tipos[nombre])
        ultimo[nombre] = (marca_ns, mensaje)
        if nombre != cmd_vel_topic:
            continue

        observaciones = (scan_topic, pose_topic, goal_topic)
        if any(topico not in ultimo for topico in observaciones):
            descartadas += 1
            continue
        if any(
            marca_ns - ultimo[topico][0] > antiguedad_maxima * 1_000_000_000
            for topico in (scan_topic, pose_topic)
        ):
            descartadas += 1
            continue

        escaneo = ultimo[scan_topic][1]
        pose = ultimo[pose_topic][1]
        objetivo = ultimo[goal_topic][1]
        marco_pose = pose.header.frame_id
        marco_objetivo = objetivo.header.frame_id
        if not marco_pose or not marco_objetivo:
            raise ValueError(
                "La pose localizada y la meta deben incluir su marco de referencia"
            )
        if marco_pose != marco_objetivo:
            raise ValueError(
                "La pose localizada y la meta deben estar en el mismo marco "
                f"(pose={marco_pose!r}, meta={marco_objetivo!r})"
            )

        pose_robot = pose.pose.pose
        pose_objetivo = objetivo.pose
        distancia, angulo = calcular_objetivo_relativo(
            pose_robot.position,
            pose_robot.orientation,
            pose_objetivo.position,
        )
        muestras.append(
            MuestraNavegacion(
                marca_temporal=(marca_ns - primera_marca_ns) / 1_000_000_000,
                id_episodio=id_episodio,
                lecturas_lidar=remuestrear_lidar(
                    escaneo.ranges, escaneo.range_min, escaneo.range_max
                ),
                distancia_objetivo=distancia,
                angulo_objetivo=angulo,
                velocidad_lineal_experta=float(mensaje.linear.x),
                velocidad_angular_experta=float(mensaje.angular.z),
            )
        )

    if not muestras:
        raise ValueError(
            "No se generaron muestras: comprueba que los tópicos tienen datos "
            "sincronizados y que el rosbag corresponde a un episodio con meta."
        )
    cantidad = guardar_jsonl(muestras, ruta_salida)
    print(f"Dataset guardado: {cantidad} muestras en {ruta_salida}")
    if descartadas:
        print(f"Comandos descartados por falta/desfase de observaciones: {descartadas}")
    return cantidad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte rosbag2 a muestras JSONL para Behavioral Cloning."
    )
    parser.add_argument("bag", help="Carpeta del rosbag2")
    parser.add_argument("output", help="Archivo JSONL de salida")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--pose-topic", default="/amcl_pose")
    parser.add_argument("--goal-topic", default="/goal_pose")
    parser.add_argument(
        "--max-age",
        type=float,
        default=0.5,
        help="Antigüedad máxima de scan/pose en segundos",
    )
    args = parser.parse_args()
    convertir(
        args.bag,
        args.output,
        scan_topic=args.scan_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        pose_topic=args.pose_topic,
        goal_topic=args.goal_topic,
        antiguedad_maxima=args.max_age,
    )


if __name__ == "__main__":
    main()
