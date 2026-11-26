
import argparse
import json

from pathlib import Path
import time
import uuid

from ament_index_python.packages import get_package_share_directory
from navil_expert.goal_sender import GoalSender
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml


def cargar_escenario(nombre):
    """Carga los objetivos de un escenario YAML."""
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
            f'Disponibles: {disponibles}'
        )

    escenario = escenarios[nombre]
    puntos = escenario.get('waypoints', [])

    if not puntos:
        raise ValueError(
            f"El escenario '{nombre}' no tiene objetivos."
        )

    waypoints = []

    for punto in puntos:
        waypoints.append((
            float(punto['x']),
            float(punto['y']),
            float(punto.get('yaw', 0.0))
        ))

    return waypoints


def publicar_evento(publicador, node, evento, episodio_id, escenario, indice, total, x, y, yaw):
    """Publica informacion sobre el estado de un episodio."""
    datos = {
        'event': evento,
        'episode_id': episodio_id,
        'scenario': escenario,
        'waypoint_index': indice,
        'waypoint_total': total,
        'stamp': node.get_clock().now().nanoseconds / 1e9,
        'goal': {
            'x': x,
            'y': y,
            'yaw_deg': yaw
        }
    }

    mensaje = String()
    mensaje.data = json.dumps(datos)

    publicador.publish(mensaje)

    node.get_logger().info(
        f'Evento publicado: {evento} ({episodio_id})'
    )


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Ejecutar rutas automaticas mediante Nav2.'
    )

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

    if argumentos.scenario:
        try:
            waypoints = cargar_escenario(argumentos.scenario)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    else:
        waypoints = argumentos.waypoint

    rclpy.init(args=ros_args)
    node = GoalSender()

    # Los eventos se publican como mensajes JSON.
    # Transient local conserva los ultimos mensajes mientras
    # este publicador siga activo.
    qos_eventos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL
    )

    publicador = node.create_publisher(
        String,
        '/navil/episode_event',
        qos_eventos
    )

    # Esperamos brevemente a que ROS 2 descubra los suscriptores.
    # Asi evitamos perder el primer evento por una conexion tardia.
    limite = time.monotonic() + 5.0

    while (
        publicador.get_subscription_count() == 0
        and time.monotonic() < limite
    ):
        rclpy.spin_once(node, timeout_sec=0.1)

    if publicador.get_subscription_count() == 0:
        node.get_logger().warn(
            'No hay suscriptores de /navil/episode_event. '
            'Los eventos podrian perderse.'
        )
    else:
        node.get_logger().info(
            'Suscriptor de episodios conectado.'
        )

    # Identificador diferente para cada ejecucion.
    ejecucion_id = uuid.uuid4().hex[:8]

    escenario = argumentos.scenario or 'manual'

    try:
        total = len(waypoints)

        node.get_logger().info(
            f'Iniciando ruta con {total} objetivos.'
        )

        for indice, (x, y, yaw) in enumerate(waypoints, start=1):

            episodio_id = f'{ejecucion_id}_{indice:03d}'

            node.get_logger().info(
                f'Objetivo {indice}/{total}: '
                f'({x:.2f}, {y:.2f}, {yaw:.1f} grados)'
            )

            # Indicamos que comienza un nuevo episodio.
            publicar_evento(
                publicador, node, 'episode_start',
                episodio_id, escenario, indice, total,
                x, y, yaw
            )

            # Reutilizamos nuestro cliente de Nav2.
            exito = node.send_goal(x, y, yaw)

            if not exito:
                publicar_evento(
                    publicador, node, 'episode_failed',
                    episodio_id, escenario, indice, total,
                    x, y, yaw
                )

                node.get_logger().error(
                    f'Ruta interrumpida en el objetivo {indice}.'
                )
                return 1

            # Nav2 ha confirmado que el objetivo se alcanzo.
            publicar_evento(
                publicador, node, 'episode_success',
                episodio_id, escenario, indice, total,
                x, y, yaw
            )

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
