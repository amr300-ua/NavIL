
import argparse
import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class GoalSender(Node):

    def __init__(self):
        super().__init__('goal_sender')

        # Cliente para enviar destinos a Nav2.
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        # Publicador para comunicar el objetivo a los otros módulos.
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/navil/goal_pose',
            10
        )       

    def send_goal(self, x, y, yaw_deg):
        self.get_logger().info('Esperando al servidor de Nav2...')

        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 no está disponible.')
            return False

        # Construimos el objetivo en el sistema de coordenadas del mapa.
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)

        # Convertimos el ángulo en grados a un cuaternión.
        yaw = math.radians(yaw_deg)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Enviamos la solicitud de navegación.
        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(
            f'Enviando objetivo: x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f}°'
        )

        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2 ha rechazado el objetivo.')
            return False

        self.get_logger().info('Objetivo aceptado por Nav2.')

        # Compartimos el objetivo con los módulos de datos y control.
        self.goal_publisher.publish(pose)
        self.get_logger().info('Objetivo publicado en /navil/goal_pose.')

        # Esperamos el resultado de la navegación.
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error('No se ha recibido el resultado.')
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Objetivo alcanzado correctamente.')
            return True

        self.get_logger().warn(
            f'Navegación no completada. Estado: {result.status}'
        )
        return False


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Enviar un objetivo de navegación a Nav2.'
    )
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--yaw', type=float, default=0.0)

    argumentos, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = GoalSender()

    try:
        node.send_goal(
            argumentos.x,
            argumentos.y,
            argumentos.yaw
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
