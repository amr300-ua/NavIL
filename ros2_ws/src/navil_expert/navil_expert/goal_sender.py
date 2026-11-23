
import rclpy
from rclpy.node import Node


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')

        self.get_logger().info('Nodo experto iniciado correctamente')


def main(args=None):
    rclpy.init(args=args)

    node = GoalSender()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
