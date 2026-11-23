
import argparse

import rclpy

from navil_expert.goal_sender import GoalSender


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Ejecutar una ruta con varios objetivos Nav2.'
    )

    parser.add_argument(
        '--waypoint',
        action='append',
        nargs=3,
        type=float,
        metavar=('X', 'Y', 'YAW'),
        required=True,
        help='Destino: coordenadas X, Y y orientacion en grados.'
    )

    argumentos, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = GoalSender()

    try:
        total = len(argumentos.waypoint)

        node.get_logger().info(
            f'Iniciando ruta con {total} objetivos.'
        )

        for indice, (x, y, yaw) in enumerate(argumentos.waypoint, start=1):
            node.get_logger().info(
                f'Objetivo {indice}/{total}: ({x:.2f}, {y:.2f}, {yaw:.1f} grados)'
            )

            exito = node.send_goal(x, y, yaw)

            if not exito:
                node.get_logger().error(
                    f'Ruta interrumpida en el objetivo {indice}.'
                )
                return 1

        node.get_logger().info(
            'Ruta completada correctamente.'
        )
        return 0

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
