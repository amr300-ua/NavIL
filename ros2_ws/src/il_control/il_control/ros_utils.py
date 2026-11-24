"""Utilidades ROS 2 compartidas por los nodos de il_control."""

import math
from typing import Optional, Tuple

from geometry_msgs.msg import Twist, TwistStamped
from rclpy.duration import Duration
from rclpy.time import Time

from il_control.observation import yaw_from_quaternion


def twist_type(stamped: bool):
    """TurtleBot 4 en Jazzy usa TwistStamped en /cmd_vel; Nav2 puede usar ambos."""
    return TwistStamped if stamped else Twist


def make_cmd(v: float, w: float, stamped: bool, stamp=None, frame_id: str = 'base_link'):
    if stamped:
        msg = TwistStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        return msg
    msg = Twist()
    msg.linear.x = float(v)
    msg.angular.z = float(w)
    return msg


def cmd_to_vw(msg) -> Tuple[float, float]:
    """(v, w) de un Twist o TwistStamped."""
    t = msg.twist if isinstance(msg, TwistStamped) else msg
    return float(t.linear.x), float(t.angular.z)


def lookup_planar(tf_buffer, target: str, source: str,
                  timeout_s: float = 0.0) -> Optional[Tuple[float, float, float]]:
    """(tx, ty, yaw) de la transformada ``target <- source`` o None si no existe.

    Se usa ``Time()`` (la más reciente disponible): para control reactivo es
    preferible una TF unos ms antigua a bloquear el bucle esperando.
    """
    try:
        tf = tf_buffer.lookup_transform(target, source, Time(),
                                        timeout=Duration(seconds=timeout_s))
    except Exception:  # tf2 lanza varias excepciones distintas
        return None
    t, q = tf.transform.translation, tf.transform.rotation
    yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in (t.x, t.y, yaw)):
        return None
    return float(t.x), float(t.y), yaw


def now_s(node) -> float:
    """Tiempo actual del reloj del nodo en segundos (respeta use_sim_time)."""
    return node.get_clock().now().nanoseconds * 1e-9
