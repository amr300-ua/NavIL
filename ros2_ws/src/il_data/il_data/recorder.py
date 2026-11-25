"""Graba los tópicos necesarios para generar demostraciones NavIL."""

import argparse
from pathlib import Path


"""
Tiene como funcion conectarse a ROS 2 y escuchar los topicos del robot.

/scan: Lidar 
/cmd_vel: Velocidades ordenadas 
/amcl_pose: Pose estimada del robot
/goal_pose: objetivo de navegacion

Utiliza rclpy para conectarse a ROS 2 y rosbag2_py para almacenar los mensajes.
"""

def crear_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graba LiDAR, velocidad experta y poses en un rosbag2."
    )
    parser.add_argument("--output", required=True, help="Carpeta del rosbag de salida")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--pose-topic", default="/amcl_pose")
    parser.add_argument("--goal-topic", default="/navil/goal_pose")
    return parser


def main() -> None:
    args = crear_argumentos().parse_args()
    ruta_salida = Path(args.output)
    if ruta_salida.exists():
        raise SystemExit(f"La ruta de salida ya existe: {ruta_salida}")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rosbag2_py import (
            ConverterOptions,
            SequentialWriter,
            StorageOptions,
            TopicMetadata,
        )
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import LaserScan
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
            especificaciones = (
                (args.scan_topic, LaserScan, qos_profile_sensor_data),
                (args.cmd_vel_topic, Twist, 10),
                (args.pose_topic, PoseWithCovarianceStamped, 10),
                (args.goal_topic, PoseStamped, 10),
            )
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

        def _escribir(self, topico: str, mensaje: object) -> None:
            self._writer.write(
                topico,
                serialize_message(mensaje),
                self.get_clock().now().nanoseconds,
            )

    rclpy.init()
    nodo = GrabadorNavIL()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
