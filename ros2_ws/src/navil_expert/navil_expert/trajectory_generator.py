
import argparse
from pathlib import Path

import yaml
import rclpy

from ament_index_python.packages import get_package_share_directory
from navil_expert.goal_sender import GoalSender


def cargar_escenario(nombre):
    """Carga los objetivos de un escenario YAML."""

    # Localizamos el archivo de configuracion del paquete ROS 2.
    ruta = (
        Path(get_package_share_directory('navil_expert'))
        / 'config'
        / 'scenarios.yaml'
    )

    with ruta.open('r', encoding='utf-8') as archivo:
        configuracion = yaml.safe_load(archivo)

    escenarios = configuracion.get('escenarios', {})

    if nombre not in escenarios:
        disponibles = ', '.join(escenarios.keys())
        raise ValueError(
            f"Escenario '{nombre}' no encontrado. "
            f"Disponibles: {disponibles}"
        )

    escenario = escenarios[nombre]
    puntos = escenario.get('waypoints', [])

    if not puntos:
        raise ValueError(
            f"El escenario '{nombre}' no tiene objetivos."
        )

    # Convertimos cada objetivo al formato (x, y, yaw).
    waypoints = []

    for punto in puntos:
        waypoints.append((
            float(punto['x']),
            float(punto['y']),
            float(punto.get('yaw', 0.0))
        ))

    return waypoints


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Ejecutar rutas automaticas mediante Nav2.'
    )

    # Solo se puede seleccionar uno de los dos modos.
    grupo = parser.add_mutually_exclusive_group(required=True)

    grupo.add_argument(
        '--waypoint',
        action='append',
        nargs=3,
        type=float,
        metavar=('X', 'Y', 'YAW'),
        help='Destino: X, Y y orientacion en grados.'
    )

    grupo.add_argument(
        '--scenario',
        type=str,
        help='Nombre del escenario definido en scenarios.yaml.'
    )

    argumentos, ros_args = parser.parse_known_args(args=args)

    # Seleccionamos de donde obtener los objetivos.
    if argumentos.scenario:
        try:
            waypoints = cargar_escenario(argumentos.scenario)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    else:
        waypoints = argumentos.waypoint

    rclpy.init(args=ros_args)
    node = GoalSender()

    try:
        total = len(waypoints)

        node.get_logger().info(
            f'Iniciando ruta con {total} objetivos.'
        )

        for indice, (x, y, yaw) in enumerate(waypoints, start=1):
            node.get_logger().info(
                f'Objetivo {indice}/{total}: '
                f'({x:.2f}, {y:.2f}, {yaw:.1f} grados)'
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
