"""Ejecuta episodios de navegación en bucle cerrado y mide su resultado.

Por qué un runner propio y no ``navil_expert/trajectory_generator``: el de
Lucas envía cada objetivo a Nav2 (acción ``NavigateToPose``). Con la política
al mando, Nav2 seguiría intentando controlar el robot, lanzaría sus
recuperaciones y decidiría él el éxito o el fallo. Este runner:

* Publica el objetivo en ``/navil/goal_pose`` y los eventos en
  ``/navil/episode_event`` con **el mismo formato JSON** que navil_expert
  (así il_data y cualquier herramienta valen para ambos).
* ``controller:=policy``: pone el mux en modo policy. El éxito se decide por
  distancia al objetivo (``goal_tolerance``).
* ``controller:=expert``: además envía el objetivo a Nav2 y pone el mux en
  modo expert; el éxito lo decide Nav2. Así Nav2 se evalúa con **las mismas
  rutas, límites y métricas** que la red (ciclo 3). Con
  ``expert_uses_nav2:=false`` no usa la acción (experto mock del stub).
* Fin del episodio: éxito, timeout, atasco (no avanza), colisión (bumper del
  Create 3 en ``/hazard_detection`` o ``/il_control/sim_collision`` del stub)
  o aborto de Nav2.
* Escribe ``episodes.csv`` y ``summary.json`` en ``output_dir/<run_tag>/``.

    ros2 run il_control episode_runner --ros-args -p scenario:=ruta_validada \\
        -p controller:=policy -p repetitions:=5 -p use_sim_time:=true
"""

import json
import math
import os
import time
import uuid

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener

from il_control.episodes import (ABORTED, EpisodeLimits, EpisodeTracker, load_scenarios,
                                 ResultsWriter, SUCCESS, waypoints_from_flat)
from il_control.ros_utils import lookup_planar, now_s

CONTROLLERS = ('policy', 'expert')


def parse_floats(text: str):
    """'1, 2, 90; 3,4,0' -> [1.0, 2.0, 90.0, 3.0, 4.0, 0.0]"""
    return [float(t) for t in text.replace(';', ',').split(',') if t.strip()]


def _yaw_quat(yaw_rad: float):
    return 0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)


class EpisodeRunner(Node):

    def __init__(self):
        super().__init__('episode_runner')
        p = self.declare_parameter
        p('scenario', '')
        p('scenarios_file', '')
        p('waypoints', '')           # alternativa a scenario: "x,y,yaw; x,y,yaw" (map, grados)
        p('controller', 'policy')
        p('repetitions', 1)
        p('output_dir', '~/navil_results')
        p('run_tag', '')
        p('policy_label', '')
        p('goal_topic', '/navil/goal_pose')
        p('event_topic', '/navil/episode_event')
        p('global_frame', 'map')
        p('odom_frame', 'odom')
        p('base_frame', 'base_link')
        p('goal_tolerance', 0.25)
        p('timeout_base', 30.0)
        p('timeout_per_meter', 10.0)
        p('stuck_window', 15.0)
        p('stuck_distance', 0.10)
        p('stop_on_collision', True)
        p('stop_on_failure', False)
        p('set_mux_mode', True)
        p('mux_node', '/cmd_vel_mux')
        p('nav2_action', '/navigate_to_pose')
        p('expert_uses_nav2', True)      # false: experto mock (fake_expert), sin acción
        p('hazard_topic', '/hazard_detection')
        p('collision_topic', '/il_control/sim_collision')
        p('scan_topic', '/scan')
        p('inference_topic', '/il_policy_node/inference_time_ms')
        p('status_topic', '/il_control/status')
        p('initial_pose', '')        # "x,y,yaw" en map -> /initialpose (AMCL); vacío = no
        p('start_delay', 2.0)
        p('pause_between', 2.0)
        p('ready_timeout', 60.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.controller = g('controller')
        if self.controller not in CONTROLLERS:
            raise ValueError(f"controller '{self.controller}' no válido: {CONTROLLERS}")
        self.scenario = g('scenario') or 'manual'
        self.waypoints = self._load_waypoints(g('scenario'), g('scenarios_file'),
                                              g('waypoints'))
        self.repetitions = max(1, int(g('repetitions')))
        self.limits = EpisodeLimits(g('goal_tolerance'), g('timeout_base'),
                                    g('timeout_per_meter'), g('stuck_window'),
                                    g('stuck_distance'))
        self.frames = (g('global_frame'), g('odom_frame'), g('base_frame'))
        self.stop_on_collision = bool(g('stop_on_collision'))
        self.stop_on_failure = bool(g('stop_on_failure'))
        self.set_mux = bool(g('set_mux_mode'))
        self.start_delay = float(g('start_delay'))
        self.pause = float(g('pause_between'))
        self.ready_timeout = float(g('ready_timeout'))

        self.run_id = uuid.uuid4().hex[:8]
        stamp = time.strftime('%Y%m%d_%H%M%S')
        tag = g('run_tag') or f'{self.controller}_{self.scenario}_{stamp}'
        self.policy_label = g('policy_label') or ('nav2' if self.controller == 'expert'
                                                  else 'unknown')
        self.writer = ResultsWriter(os.path.join(os.path.expanduser(g('output_dir')), tag), {
            'run_id': self.run_id, 'run_tag': tag, 'controller': self.controller,
            'policy': self.policy_label, 'scenario': self.scenario,
            'waypoints': self.waypoints, 'repetitions': self.repetitions,
            'limits': vars(self.limits), 'started': stamp})

        # --- comunicaciones
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(PoseStamped, g('goal_topic'), 10)
        self.event_pub = self.create_publisher(String, g('event_topic'), 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(LaserScan, g('scan_topic'), self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32, g('inference_topic'), self._on_latency, 10)
        self.create_subscription(String, g('status_topic'), self._on_status, 10)
        self.create_subscription(Bool, g('collision_topic'),
                                 lambda m: m.data and self._on_collision('sim'), 10)
        self._setup_hazards(g('hazard_topic'))
        self.mux_client = self.create_client(SetParameters, f"{g('mux_node')}/set_parameters")
        use_nav2 = self.controller == 'expert' and bool(g('expert_uses_nav2'))
        self.nav2 = self._setup_nav2(g('nav2_action')) if use_nav2 else None

        # --- estado
        self.queue = [(rep, i, wp) for rep in range(1, self.repetitions + 1)
                      for i, wp in enumerate(self.waypoints, start=1)]
        self.k = 0
        self.tracker = None
        self.state = 'init'
        self.state_t = None
        self.transitions = {}
        self._estop_snapshot = 0
        self._last_hazard = -math.inf
        self.nav2_goal_handle = None
        self.nav2_outcome = None
        self.done = Future()

        init = parse_floats(g('initial_pose'))
        if init and len(init) != 3:
            raise ValueError('initial_pose debe ser "x,y,yaw_deg"')
        self.initial_pose = init or None
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f'Runner {self.run_id}: {self.controller} | escenario {self.scenario} | '
            f'{len(self.waypoints)} objetivos x {self.repetitions} repeticiones -> '
            f'{self.writer.dir}')

    # -------------------------------------------------------------- utilidades
    def _load_waypoints(self, scenario, scenarios_file, flat):
        if flat:
            return waypoints_from_flat(parse_floats(flat))
        if not scenario:
            raise ValueError('Indica scenario o waypoints')
        path = scenarios_file or os.path.join(
            get_package_share_directory('il_control'), 'config', 'scenarios.yaml')
        scenarios = load_scenarios(path)
        if scenario not in scenarios:
            raise ValueError(f"Escenario '{scenario}' no está en {path}. "
                             f'Disponibles: {", ".join(scenarios)}')
        return scenarios[scenario]

    def _setup_hazards(self, topic):
        """Bumper del Create 3 (solo si irobot_create_msgs está instalado)."""
        try:
            from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
        except ImportError:
            self.get_logger().warn('irobot_create_msgs no disponible: colisiones solo por '
                                   'collision_topic')
            return
        bump = HazardDetection.BUMP

        def cb(msg):
            if any(d.type == bump for d in msg.detections):
                self._on_collision('bumper')
        self.create_subscription(HazardDetectionVector, topic, cb, qos_profile_sensor_data)

    def _setup_nav2(self, action_name):
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        self._NavigateToPose = NavigateToPose
        return ActionClient(self, NavigateToPose, action_name)

    def _pose(self, target, source):
        return lookup_planar(self.tf_buffer, target, source)

    def _event(self, name, rep, idx, wp, **extra):
        data = {'event': name, 'episode_id': f'{self.run_id}_{self.k + 1:03d}',
                'scenario': self.scenario, 'waypoint_index': idx,
                'waypoint_total': len(self.waypoints), 'stamp': now_s(self),
                'goal': {'x': wp[0], 'y': wp[1], 'yaw_deg': wp[2]},
                'controller': self.controller, 'repetition': rep, 'source': 'il_control'}
        data.update(extra)
        self.event_pub.publish(String(data=json.dumps(data)))

    def _set_mux(self, mode):
        if not self.set_mux:
            return
        if not self.mux_client.service_is_ready():
            self.get_logger().error(f'Servicio del mux no disponible: no se puede poner {mode}')
            return
        req = SetParameters.Request()
        req.parameters = [Parameter('mode', Parameter.Type.STRING, mode).to_parameter_msg()]
        self.mux_client.call_async(req)

    def _goto(self, state):
        self.state, self.state_t = state, now_s(self)

    # ---------------------------------------------------------------- entradas
    def _on_scan(self, msg: LaserScan):
        if self.tracker is None:
            return
        valid = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        if valid:
            self.tracker.add_scan_min(min(valid))

    def _on_latency(self, msg: Float32):
        if self.tracker is not None:
            self.tracker.add_latency(msg.data)

    def _on_status(self, msg: String):
        try:
            self.transitions = json.loads(msg.data).get('transitions', {})
        except (ValueError, AttributeError):
            pass

    def _estops(self):
        return sum(v for k, v in self.transitions.items() if k.startswith('estop'))

    def _on_collision(self, source):
        t = now_s(self)
        if self.tracker is not None and t - self._last_hazard > 2.0:  # un choque = un evento
            self.tracker.add_collision()
            self.get_logger().error(f'Colisión detectada ({source})')
        self._last_hazard = t

    # ------------------------------------------------------------ máquina de estados
    def _tick(self):
        t = now_s(self)
        if self.state_t is None:
            self.state_t = t
        handler = getattr(self, f'_st_{self.state}')
        handler(t, t - self.state_t)

    def _st_init(self, t, dt):
        if self.initial_pose:
            x, y, yaw = self.initial_pose
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id, msg.header.stamp = self.frames[0], self.get_clock().now().to_msg()
            msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
            o = msg.pose.pose.orientation
            o.x, o.y, o.z, o.w = _yaw_quat(math.radians(yaw))
            msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.07
            self.init_pub.publish(msg)
            self.get_logger().info(f'Pose inicial publicada en /initialpose: {self.initial_pose}')
        self._goto('wait_ready')

    def _st_wait_ready(self, t, dt):
        g, o, b = self.frames
        missing = []
        if self._pose(g, b) is None:
            missing.append(f'TF {g}<-{b}')
        if self._pose(o, b) is None:
            missing.append(f'TF {o}<-{b}')
        if self.set_mux and not self.mux_client.service_is_ready():
            missing.append('mux')
        if self.nav2 is None and self.goal_pub.get_subscription_count() == 0:
            missing.append('suscriptor del objetivo (il_policy_node)')
        if self.nav2 is not None and not self.nav2.server_is_ready():
            missing.append('Nav2')
        if not missing and dt >= self.start_delay:
            self._goto('start')
        elif dt > self.ready_timeout:
            self.get_logger().fatal(f'Sin {", ".join(missing)} tras {self.ready_timeout:.0f} s')
            self._finish()
        elif missing:
            self.get_logger().info(f'Esperando: {", ".join(missing)}', throttle_duration_sec=5.0)

    def _st_start(self, t, dt):
        rep, idx, wp = self.queue[self.k]
        pose = self._pose(self.frames[0], self.frames[2])
        if pose is None:
            return
        self.tracker = EpisodeTracker((wp[0], wp[1]), t, (pose[0], pose[1]), self.limits)
        self._estop_snapshot = self._estops()
        self.nav2_outcome = None
        self._set_mux(self.controller)
        self._event('episode_start', rep, idx, wp)

        goal = PoseStamped()
        goal.header.frame_id, goal.header.stamp = self.frames[0], self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = wp[0], wp[1]
        o = goal.pose.orientation
        o.x, o.y, o.z, o.w = _yaw_quat(math.radians(wp[2]))
        self.goal_pub.publish(goal)
        if self.nav2 is not None:
            self._send_nav2(goal)
        self.get_logger().info(
            f'[{self.k + 1}/{len(self.queue)}] rep {rep} objetivo {idx}: ({wp[0]:.2f}, '
            f'{wp[1]:.2f}) a {self.tracker.straight:.2f} m, timeout {self.tracker.timeout:.0f} s')
        self._goto('running')

    def _send_nav2(self, pose):
        goal = self._NavigateToPose.Goal()
        goal.pose = pose
        fut = self.nav2.send_goal_async(goal)

        def accepted(f):
            h = f.result()
            if h is None or not h.accepted:
                self.nav2_outcome = ABORTED
                return
            self.nav2_goal_handle = h
            h.get_result_async().add_done_callback(done)

        def done(f):
            from action_msgs.msg import GoalStatus
            res = f.result()
            ok = res is not None and res.status == GoalStatus.STATUS_SUCCEEDED
            self.nav2_outcome = SUCCESS if ok else ABORTED
        fut.add_done_callback(accepted)

    def _st_running(self, t, dt):
        g, o, b = self.frames
        mp, od = self._pose(g, b), self._pose(o, b)
        if od is not None:
            self.tracker.update_odom(od[0], od[1])
        if mp is not None:
            self.tracker.update_map(t, mp[0], mp[1])
        outcome = self.tracker.check(t, self.stop_on_collision,
                                     use_distance_success=self.nav2 is None)
        if outcome is None and self.nav2_outcome is not None:
            outcome = self.nav2_outcome
        if outcome is not None:
            self._end_episode(outcome)

    def _end_episode(self, outcome):
        rep, idx, wp = self.queue[self.k]
        if self.nav2_goal_handle is not None and outcome != SUCCESS:
            self.nav2_goal_handle.cancel_goal_async()
        self.nav2_goal_handle = None
        self._set_mux('stop')
        self.tracker.estop_events = self._estops() - self._estop_snapshot
        res = self.tracker.result(
            outcome, episode_id=f'{self.run_id}_{self.k + 1:03d}', run_id=self.run_id,
            scenario=self.scenario, controller=self.controller, policy=self.policy_label,
            repetition=rep, waypoint_index=idx, waypoint_total=len(self.waypoints))
        self.writer.add(res)
        ev = 'episode_success' if res.success else 'episode_failed'
        self._event(ev, rep, idx, wp, outcome=outcome, metrics=res.to_row())
        msg = (f'Episodio {res.episode_id}: {outcome} | {res.time_s:.1f} s | '
               f'{res.path_length_m:.2f} m | colisiones {res.collisions} | '
               f'e-stops {res.estop_events} | d_final {res.final_dist_m:.2f} m')
        if res.success:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg)

        self.tracker = None
        self.k += 1
        if self.k >= len(self.queue) or (not res.success and self.stop_on_failure):
            self._finish()
        else:
            self._goto('pause')

    def _st_pause(self, t, dt):
        if dt >= self.pause:
            self._goto('start')

    def _st_done(self, t, dt):
        pass

    def _finish(self):
        self._set_mux('stop')
        s = self.writer.write_summary()['summary']
        self.get_logger().info(
            f"FIN {self.run_id}: éxito {s['successes']}/{s['episodes']} | "
            f"colisiones {s['collisions_total']} | resultados en {self.writer.dir}")
        self._goto('done')
        if not self.done.done():
            self.done.set_result(True)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EpisodeRunner()
        rclpy.spin_until_future_complete(node, node.done)
        rclpy.spin_once(node, timeout_sec=0.5)  # deja salir la última petición al mux
    except KeyboardInterrupt:
        if node is not None and node.writer.results:
            node.writer.write_summary()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
