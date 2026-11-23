"""Simulador cinemático mínimo (stub de Gazebo + TurtleBot 4) para el ciclo 1.

Publica lo mismo que el robot real necesita para il_control:
``/scan`` (LaserScan en ``rplidar_link``), ``/odom``, TF ``odom -> base_link``,
TF estáticas ``map -> odom`` y ``base_link -> rplidar_link`` (con yaw
configurable para emular un LiDAR montado girado) y un ``/goal_pose``.
Integra lo que llega a ``/cmd_vel``. Si el robot toca un obstáculo se detiene
y publica ``/il_control/sim_collision``.
"""

import math

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from il_control.ros_utils import cmd_to_vw, now_s, twist_type
from il_control.sim_world import SimWorld, integrate_unicycle


def _quat_z(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _tf(stamp, parent, child, x, y, yaw):
    t = TransformStamped()
    t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
    t.transform.translation.x, t.transform.translation.y = float(x), float(y)
    q = _quat_z(yaw)
    (t.transform.rotation.x, t.transform.rotation.y,
     t.transform.rotation.z, t.transform.rotation.w) = q
    return t


class FakeSim(Node):

    def __init__(self):
        super().__init__('fake_sim')
        p = self.declare_parameter
        p('rate', 50.0)
        p('scan_rate', 10.0)
        p('cmd_topic', '/cmd_vel')
        p('cmd_stamped', True)
        p('cmd_timeout', 0.5)
        p('n_rays', 360)
        p('range_min', 0.15)
        p('range_max', 12.0)
        p('noise_std', 0.0)
        p('lidar_frame', 'rplidar_link')
        p('lidar_yaw', 0.0)
        p('room_half', 4.0)
        p('obstacles', [1.5, 0.0, 0.3])
        p('start', [-3.0, 0.0, 0.0])
        p('goal', [3.0, 0.0])
        p('goal_delay', 3.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.world = SimWorld(g('room_half'), list(g('obstacles')))
        self.x, self.y, self.yaw = (float(v) for v in g('start'))
        self.v = self.w = 0.0
        self.cmd_t = None
        self.collided = False
        self.cmd_timeout = float(g('cmd_timeout'))
        self.n_rays = int(g('n_rays'))
        self.range_min, self.range_max = float(g('range_min')), float(g('range_max'))
        self.noise = float(g('noise_std'))
        self.lidar_frame, self.lidar_yaw = g('lidar_frame'), float(g('lidar_yaw'))
        self.goal = [float(v) for v in g('goal')]
        self.rng = np.random.default_rng(0)
        self.dt = 1.0 / float(g('rate'))

        self.create_subscription(twist_type(g('cmd_stamped')), g('cmd_topic'), self._on_cmd, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)
        self.coll_pub = self.create_publisher(Bool, '/il_control/sim_collision', 10)
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        self.static_tf.sendTransform([
            _tf(stamp, 'map', 'odom', 0.0, 0.0, 0.0),
            _tf(stamp, 'base_link', self.lidar_frame, 0.0, 0.0, self.lidar_yaw)])

        self.create_timer(self.dt, self._step)
        self.create_timer(1.0 / float(g('scan_rate')), self._publish_scan)
        self._goal_timer = self.create_timer(float(g('goal_delay')), self._publish_goal)
        self.get_logger().info(
            f'Sim: robot en ({self.x:.1f}, {self.y:.1f}), objetivo {self.goal}, '
            f'LiDAR girado {math.degrees(self.lidar_yaw):.0f}°')

    def _on_cmd(self, msg):
        self.v, self.w = cmd_to_vw(msg)
        self.cmd_t = now_s(self)

    def _step(self):
        if self.cmd_t is None or now_s(self) - self.cmd_t > self.cmd_timeout:
            self.v = self.w = 0.0  # como la base real: sin comandos, se para
        if not self.collided:
            x, y, yaw = integrate_unicycle(self.x, self.y, self.yaw, self.v, self.w, self.dt)
            if self.world.in_collision(x, y):
                self.collided = True
                self.coll_pub.publish(Bool(data=True))
                self.get_logger().error(f'¡COLISIÓN en ({x:.2f}, {y:.2f})! Robot detenido.')
            else:
                self.x, self.y, self.yaw = x, y, yaw
        stamp = self.get_clock().now().to_msg()
        self.tf_pub.sendTransform(_tf(stamp, 'odom', 'base_link', self.x, self.y, self.yaw))
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        q = _quat_z(self.yaw)
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        moving = not self.collided
        odom.twist.twist.linear.x = self.v if moving else 0.0
        odom.twist.twist.angular.z = self.w if moving else 0.0
        self.odom_pub.publish(odom)

    def _publish_scan(self):
        inc = 2.0 * math.pi / self.n_rays
        angles = -math.pi + np.arange(self.n_rays) * inc
        r = self.world.raycast(self.x, self.y, self.yaw + self.lidar_yaw, angles,
                               self.range_max)
        if self.noise > 0:
            r = r + self.rng.normal(0.0, self.noise, r.shape)
        r = np.where(np.isfinite(r) & (r < self.range_min), -np.inf, r)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.lidar_frame
        msg.angle_min, msg.angle_increment = -math.pi, inc
        msg.angle_max = -math.pi + (self.n_rays - 1) * inc
        msg.range_min, msg.range_max = self.range_min, self.range_max
        msg.ranges = [float(v) for v in r]
        self.scan_pub.publish(msg)

    def _publish_goal(self):
        self._goal_timer.cancel()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x, msg.pose.position.y = self.goal
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f'Objetivo publicado en /goal_pose: {self.goal}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FakeSim()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
