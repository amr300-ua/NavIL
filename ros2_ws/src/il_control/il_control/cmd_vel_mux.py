"""Selector experto/política + capa de seguridad. Único publicador de /cmd_vel.

Por qué un nodo propio y no ``twist_mux``:
* ``twist_mux`` elige por prioridad y por "quién publicó último". Aquí se
  necesita **exclusión explícita** (o Nav2 o la red, nunca ambos), elegida por
  el experimento, y que la fuente inactiva se ignore aunque siga publicando
  (la política puede correr en modo sombra).
* La seguridad tiene que aplicarse **después** de elegir la fuente y justo
  antes del robot, para que ningún controlador pueda saltársela.

Cambio de modo en caliente::

    ros2 param set /cmd_vel_mux mode policy   # expert | policy | stop

Publica a frecuencia fija (``publish_rate``) aunque las entradas se detengan:
así el watchdog puede mandar ceros activamente.
"""

import json

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from il_control.observation import scan_to_points
from il_control.ros_utils import cmd_to_vw, lookup_planar, make_cmd, now_s, twist_type
from il_control.safety import MODE_STOP, SafetyConfig, SafetyFilter

MODES = ('expert', 'policy', 'stop')
SAFETY_PARAMS = ('v_max', 'v_min', 'w_max', 'max_lin_accel', 'max_ang_accel',
                 'robot_radius', 'lateral_margin', 'stop_distance', 'slow_distance',
                 'scan_timeout', 'cmd_timeout', 'proximity_check')


class CmdVelMux(Node):

    def __init__(self):
        super().__init__('cmd_vel_mux')
        p = self.declare_parameter
        p('mode', 'expert')
        p('expert_topic', '/cmd_vel_expert')
        p('policy_topic', '/cmd_vel_policy')
        p('output_topic', '/cmd_vel')
        p('expert_stamped', True)
        p('policy_stamped', True)
        p('output_stamped', True)
        p('scan_topic', '/scan')
        p('base_frame', 'base_link')
        p('use_lidar_tf', True)
        p('lidar_yaw_offset', 0.0)
        p('publish_rate', 20.0)
        p('proximity_check_expert', True)
        defaults = SafetyConfig()
        for name in SAFETY_PARAMS:
            p(name, getattr(defaults, name))
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.mode = g('mode')
        if self.mode not in MODES:
            raise ValueError(f"mode '{self.mode}' no válido: {MODES}")
        self.safety = SafetyFilter(SafetyConfig(**{n: g(n) for n in SAFETY_PARAMS}))
        self.base_frame = g('base_frame')
        self.out_stamped = bool(g('output_stamped'))
        self.prox_expert = bool(g('proximity_check_expert'))
        self.lidar_tf = None if g('use_lidar_tf') else (0.0, 0.0, float(g('lidar_yaw_offset')))
        self.lidar_yaw_fallback = float(g('lidar_yaw_offset'))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last = {'expert': None, 'policy': None}  # (v, w, t)
        self.points = None
        self.scan_t = None
        self.reason = None
        self.counts = {}

        self.create_subscription(twist_type(g('expert_stamped')), g('expert_topic'),
                                 lambda m: self._on_cmd('expert', m), 10)
        self.create_subscription(twist_type(g('policy_stamped')), g('policy_topic'),
                                 lambda m: self._on_cmd('policy', m), 10)
        self.create_subscription(LaserScan, g('scan_topic'), self._on_scan,
                                 qos_profile_sensor_data)
        self.out_pub = self.create_publisher(twist_type(self.out_stamped),
                                             g('output_topic'), 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, '/il_control/mode', latched)
        self.status_pub = self.create_publisher(String, '/il_control/status', 10)

        self.add_on_set_parameters_callback(self._on_params)
        self.create_timer(1.0 / float(g('publish_rate')), self._tick)
        self.create_timer(0.5, self._publish_status)
        self.mode_pub.publish(String(data=self.mode))
        self.get_logger().info(
            f"Mux activo | modo={self.mode} | experto={g('expert_topic')} | "
            f"política={g('policy_topic')} -> {g('output_topic')} "
            f"({'TwistStamped' if self.out_stamped else 'Twist'})")

    # ---------------------------------------------------------------- entradas
    def _on_cmd(self, source: str, msg):
        v, w = cmd_to_vw(msg)
        self.last[source] = (v, w, now_s(self))

    def _on_scan(self, msg: LaserScan):
        if self.lidar_tf is None:
            tf = lookup_planar(self.tf_buffer, self.base_frame, msg.header.frame_id)
            if tf is not None:
                self.lidar_tf = tf
                self.get_logger().info(f'TF del LiDAR: yaw={tf[2]:+.3f} rad, '
                                       f'offset=({tf[0]:.3f}, {tf[1]:.3f}) m')
        x, y, yaw = self.lidar_tf or (0.0, 0.0, self.lidar_yaw_fallback)
        self.points = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                     msg.range_min, msg.range_max, yaw, x, y)
        self.scan_t = now_s(self)

    def _on_params(self, params):
        for prm in params:
            if prm.name == 'mode':
                if prm.value not in MODES:
                    return SetParametersResult(successful=False,
                                               reason=f'mode debe ser uno de {MODES}')
                if prm.value != self.mode:
                    self.get_logger().warn(f'Cambio de modo: {self.mode} -> {prm.value}')
                    self.mode = prm.value
                    # Se descarta el último comando de la fuente nueva: solo
                    # cuenta lo que publique a partir de ahora.
                    if self.mode in self.last:
                        self.last[self.mode] = None
                    self.safety.reset()
                    self.mode_pub.publish(String(data=self.mode))
            elif prm.name in SAFETY_PARAMS or prm.name.endswith('_topic'):
                return SetParametersResult(
                    successful=False, reason=f'{prm.name} solo se puede fijar al arrancar')
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ salida
    def _tick(self):
        now = now_s(self)
        if self.mode == 'stop':
            res = self.safety.stop(now, MODE_STOP)
        else:
            last = self.last[self.mode]
            v, w, t = last if last is not None else (0.0, 0.0, None)
            prox = True if self.mode == 'policy' else self.prox_expert
            res = self.safety.filter(v, w, now, t, self.scan_t, self.points, prox)

        if res.reason != self.reason:
            level = self.get_logger().warn if res.is_stop else self.get_logger().info
            level(f'[{self.mode}] seguridad: {self.reason} -> {res.reason}')
            self.counts[res.reason] = self.counts.get(res.reason, 0) + 1
            self.reason = res.reason
        self.out_pub.publish(make_cmd(res.v, res.w, self.out_stamped,
                                      self.get_clock().now().to_msg(), self.base_frame))
        self._last_res = res

    def _publish_status(self):
        res = getattr(self, '_last_res', None)
        if res is None:
            return
        clearance = res.clearance if res.clearance != float('inf') else None
        self.status_pub.publish(String(data=json.dumps({
            'mode': self.mode, 'reason': res.reason, 'v': round(res.v, 3),
            'w': round(res.w, 3), 'clearance': clearance,
            'transitions': self.counts})))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CmdVelMux()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            # Última orden explícita de parada antes de salir (si el contexto
            # sigue vivo; con Ctrl+C rclpy puede haberlo cerrado ya).
            try:
                node.out_pub.publish(make_cmd(0.0, 0.0, node.out_stamped,
                                              node.get_clock().now().to_msg(),
                                              node.base_frame))
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
