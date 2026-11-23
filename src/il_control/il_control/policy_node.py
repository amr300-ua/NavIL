"""Nodo de inferencia: sensores -> observación -> política -> /cmd_vel_policy.

* Bucle de control con **timer a frecuencia fija** (``control_rate``), no en
  el callback del LiDAR: el periodo de control no depende de la frecuencia del
  sensor y la latencia es predecible.
* Este nodo **nunca publica en /cmd_vel**. Publica en ``/cmd_vel_policy`` y
  el ``cmd_vel_mux`` decide qué fuente llega al robot. Así la política puede
  ejecutarse en *modo sombra* mientras conduce Nav2 (para medir MAE online).
* Diagnóstico en tópicos privados (``~/inference_time_ms``, ``~/goal_reached``,
  ``~/observation``), para poder lanzar varias instancias (p. ej. experto mock).
* Si faltan datos (sin LiDAR reciente, sin objetivo o sin TF) no publica nada:
  el watchdog del mux lo detecta como ``cmd_timeout`` y para el robot.
"""

import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray
from tf2_ros import Buffer, TransformListener

from il_control.observation import (goal_in_base_frame, laserscan_to_sectors,
                                    transform_point_2d)
from il_control.policies import POLICY_TYPES, make_policy
from il_control.ros_utils import lookup_planar, make_cmd, now_s, twist_type


class PolicyNode(Node):

    def __init__(self):
        super().__init__('il_policy_node')
        p = self.declare_parameter
        p('policy_type', 'mock_reactive')
        p('model_path', '')
        p('device', 'cpu')
        p('control_rate', 10.0)
        p('scan_topic', '/scan')
        p('goal_topic', '/goal_pose')
        p('output_topic', '/cmd_vel_policy')
        p('output_stamped', True)
        p('base_frame', 'base_link')
        p('use_lidar_tf', True)
        p('lidar_yaw_offset', 0.0)
        p('scan_timeout', 0.5)
        p('goal_tolerance', 0.25)
        p('publish_observation', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.base_frame = g('base_frame')
        self.use_lidar_tf = g('use_lidar_tf')
        self.lidar_yaw = float(g('lidar_yaw_offset'))
        self.scan_timeout = float(g('scan_timeout'))
        self.goal_tol = float(g('goal_tolerance'))
        self.stamped = bool(g('output_stamped'))
        self.publish_obs = bool(g('publish_observation'))

        ptype = g('policy_type')
        if ptype not in POLICY_TYPES:
            raise ValueError(f"policy_type '{ptype}' no válido: {POLICY_TYPES}")
        self.policy = make_policy(ptype, g('model_path'), g('device'))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan = None
        self.scan_t = None
        self.goal = None
        self.goal_reached = False
        self._lidar_tf_ok = not self.use_lidar_tf

        self.create_subscription(LaserScan, g('scan_topic'), self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, g('goal_topic'), self._on_goal,
                                 QoSProfile(depth=1))
        self.cmd_pub = self.create_publisher(twist_type(self.stamped), g('output_topic'), 10)
        self.lat_pub = self.create_publisher(Float32, '~/inference_time_ms', 10)
        self.reached_pub = self.create_publisher(Bool, '~/goal_reached', 10)
        self.obs_pub = self.create_publisher(Float32MultiArray, '~/observation', 10)

        rate = float(g('control_rate'))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'Política: {self.policy.describe()} | {rate:.1f} Hz | '
            f"salida {g('output_topic')} ({'TwistStamped' if self.stamped else 'Twist'}) | "
            f'scan -> {self.policy.scan_size} sectores')

    # ---------------------------------------------------------------- entradas
    def _on_scan(self, msg: LaserScan):
        self.scan, self.scan_t = msg, now_s(self)
        if not self._lidar_tf_ok:
            tf = lookup_planar(self.tf_buffer, self.base_frame, msg.header.frame_id)
            if tf is not None:
                self.lidar_yaw = tf[2]
                self._lidar_tf_ok = True
                self.get_logger().info(
                    f"Yaw de '{msg.header.frame_id}' respecto a '{self.base_frame}': "
                    f'{self.lidar_yaw:+.3f} rad (desde TF)')

    def _on_goal(self, msg: PoseStamped):
        self.goal = msg
        self.goal_reached = False
        p = msg.pose.position
        self.get_logger().info(
            f"Nuevo objetivo ({p.x:.2f}, {p.y:.2f}) en '{msg.header.frame_id}'")

    # ------------------------------------------------------------------ bucle
    def _relative_goal(self):
        frame = self.goal.header.frame_id or self.base_frame
        tf = lookup_planar(self.tf_buffer, self.base_frame, frame)
        if tf is None:
            return None
        p = self.goal.pose.position
        gx, gy = transform_point_2d(*tf, p.x, p.y)
        return goal_in_base_frame(gx, gy)

    def _publish(self, v: float, w: float):
        self.cmd_pub.publish(make_cmd(v, w, self.stamped, self.get_clock().now().to_msg(),
                                      self.base_frame))

    def _tick(self):
        log = self.get_logger()
        if self.scan is None or now_s(self) - self.scan_t > self.scan_timeout:
            log.warn('Sin LiDAR reciente: no se publica comando', throttle_duration_sec=2.0)
            return
        if self.goal is None:
            log.info('Esperando objetivo...', throttle_duration_sec=5.0)
            return
        if not self._lidar_tf_ok:
            log.warn('Sin TF del LiDAR todavía; usando lidar_yaw_offset',
                     throttle_duration_sec=5.0)
        goal = self._relative_goal()
        if goal is None:
            log.warn(f"Sin TF '{self.base_frame}' <- '{self.goal.header.frame_id}'",
                     throttle_duration_sec=2.0)
            return

        if goal[0] < self.goal_tol:
            if not self.goal_reached:
                self.goal_reached = True
                self.reached_pub.publish(Bool(data=True))
                log.info(f'Objetivo alcanzado (d={goal[0]:.2f} m)')
            self._publish(0.0, 0.0)
            return

        s = self.scan
        try:
            scan90 = laserscan_to_sectors(
                s.ranges, s.angle_min, s.angle_increment, s.range_min, s.range_max,
                self.policy.scan_size, self.policy.max_range, self.lidar_yaw)
            obs = {'scan': scan90, 'goal': [goal[0], goal[1]]}
            t0 = time.perf_counter()
            v, w = self.policy.predict(obs)
            self.lat_pub.publish(Float32(data=(time.perf_counter() - t0) * 1e3))
        except Exception as e:  # el robot nunca debe quedarse con el último comando
            log.error(f'Fallo en la inferencia: {e!r}. Publicando parada.',
                      throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)
            return

        self._publish(v, w)
        if self.publish_obs:
            self.obs_pub.publish(Float32MultiArray(
                data=[float(x) for x in scan90] + [float(goal[0]), float(goal[1]),
                                                   float(v), float(w)]))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyNode()
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
