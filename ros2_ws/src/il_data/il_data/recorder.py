"""Graba los tópicos necesarios para generar demostraciones NavIL."""

import argparse
import json
from pathlib import Path

from .schema import normalizar_id_episodio


"""
Tiene como funcion conectarse a ROS 2 y escuchar los topicos del robot.

/scan: Lidar 
/cmd_vel_nav: acción publicada por el controlador Nav2
/navil/goal_pose: objetivo aceptado por el experto
/tf y /tf_static: pose del robot y orientación del LiDAR

Utiliza rclpy para conectarse a ROS 2 y rosbag2_py para almacenar los mensajes.
"""

def crear_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graba LiDAR, velocidad experta y poses en un rosbag2."
    )
    parser.add_argument("--output", required=True, help="Carpeta del rosbag de salida")
    parser.add_argument(
        "--episode-id",
        required=True,
        help="ID de Lucas, p. ej. 33f23ed2_001, o un entero",
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument(
        "--cmd-vel-topic",
        default="auto",
        help=(
            "auto graba /cmd_vel_expert y /cmd_vel_nav; puedes indicar un "
            "tópico concreto. Nunca se graba /cmd_vel"
        ),
    )
    parser.add_argument(
        "--cmd-vel-type",
        choices=("twist", "twist-stamped"),
        default="twist-stamped",
        help="Tipo de mensaje publicado por el tópico experto",
    )
    parser.add_argument(
        "--goal-topic",
        default="auto",
        help="auto graba /navil/goal_pose y /goal_pose para resolver el alias",
    )
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    return parser


def main() -> None:
    args, ros_args = crear_argumentos().parse_known_args()
    try:
        dataset_episode_id, source_episode_id = normalizar_id_episodio(
            args.episode_id
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    ruta_salida = Path(args.output)
    ruta_metadatos = ruta_salida.with_name(ruta_salida.name + ".episode.json")
    if ruta_salida.exists():
        raise SystemExit(f"La ruta de salida ya existe: {ruta_salida}")
    if ruta_metadatos.exists():
        raise SystemExit(f"Los metadatos de salida ya existen: {ruta_metadatos}")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rclpy
        from geometry_msgs.msg import (
            PoseStamped,
            Twist,
            TwistStamped,
        )
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile,
            ReliabilityPolicy,
            DurabilityPolicy,
            qos_profile_sensor_data,
        )
        from rosbag2_py import (
            ConverterOptions,
            SequentialWriter,
            StorageOptions,
            TopicMetadata,
        )
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import LaserScan
        from tf2_msgs.msg import TFMessage
    except ImportError as error:
        raise SystemExit(
            "El grabador debe ejecutarse en un entorno con ROS 2 y rosbag2_py."
        ) from error

    class GrabadorNavIL(Node):
        def __init__(self) -> None:
            super().__init__("navil_data_recorder")
            self._writer = SequentialWriter()
            self._writer.open(
                StorageOptions(uri=str(ruta_salida), storage_id="sqlite3"),
                ConverterOptions(
                    input_serialization_format="cdr",
                    output_serialization_format="cdr",
                ),
            )
            self._suscripciones = []
            self._message_counts = {}
            tipo_velocidad = (
                Twist if args.cmd_vel_type == "twist" else TwistStamped
            )
            topicos_velocidad = (
                ("/cmd_vel_expert", "/cmd_vel_nav")
                if args.cmd_vel_topic == "auto"
                else (args.cmd_vel_topic,)
            )
            topicos_objetivo = (
                ("/navil/goal_pose", "/goal_pose")
                if args.goal_topic == "auto"
                else (args.goal_topic,)
            )
            especificaciones = [
                (args.scan_topic, LaserScan, qos_profile_sensor_data),
                *((topico, tipo_velocidad, 10) for topico in topicos_velocidad),
                *((topico, PoseStamped, 10) for topico in topicos_objetivo),
                (args.tf_topic, TFMessage, 10),
            ]
            qos_tf_static = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            especificaciones.append((args.tf_static_topic, TFMessage, qos_tf_static))
            for topico, tipo, qos in especificaciones:
                self._writer.create_topic(
                    TopicMetadata(
                        name=topico,
                        type=f"{tipo.__module__.split('.')[0]}/msg/{tipo.__name__}",
                        serialization_format="cdr",
                    )
                )
                self._suscripciones.append(
                    self.create_subscription(
                        tipo,
                        topico,
                        lambda mensaje, nombre=topico: self._escribir(
                            nombre, mensaje
                        ),
                        qos,
                    )
                )
            self.get_logger().info(f"Grabando rosbag en {ruta_salida}")
            self.get_logger().info(
                f"Acciones candidatas: {topicos_velocidad} "
                f"({args.cmd_vel_type}) | metas candidatas: {topicos_objetivo} | "
                f"use_sim_time="
                f"{self.get_parameter('use_sim_time').value}"
            )

        def _escribir(self, topico: str, mensaje: object) -> None:
            self._writer.write(
                topico,
                serialize_message(mensaje),
                self.get_clock().now().nanoseconds,
            )
            self._message_counts[topico] = (
                self._message_counts.get(topico, 0) + 1
            )

    rclpy.init(args=ros_args)
    nodo = GrabadorNavIL()
    inicio_ns = nodo.get_clock().now().nanoseconds
    with ruta_metadatos.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "episode_id": source_episode_id,
                "source_episode_id": source_episode_id,
                "dataset_episode_id": dataset_episode_id,
                "requested_cmd_vel_topic": args.cmd_vel_topic,
                "requested_goal_topic": args.goal_topic,
                "bag": ruta_salida.name,
                "outcome": "unknown",
                "use_sim_time": bool(
                    nodo.get_parameter("use_sim_time").value
                ),
                "start_time_ns": inicio_ns,
            },
            archivo,
            indent=2,
        )
        archivo.write("\n")
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo._writer.close()
        fin_ns = nodo.get_clock().now().nanoseconds
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        with ruta_metadatos.open("r", encoding="utf-8") as archivo:
            metadatos = json.load(archivo)
        metadatos["end_time_ns"] = fin_ns
        metadatos["duration_seconds"] = max(0.0, (fin_ns - inicio_ns) / 1e9)
        metadatos["message_counts"] = nodo._message_counts
        with ruta_metadatos.open("w", encoding="utf-8") as archivo:
            json.dump(metadatos, archivo, indent=2)
            archivo.write("\n")


if __name__ == "__main__":
    main()
