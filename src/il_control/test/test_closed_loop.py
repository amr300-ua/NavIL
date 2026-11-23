"""Bucle cerrado sin ROS: sim stub -> preprocesado -> política -> seguridad -> robot.

Reproduce en Python puro la cadena de mock_test.launch.py (mismo mundo que
config/mock_sim.yaml, LiDAR girado 90°) para comprobar el comportamiento
del ciclo 1 en CI sin Gazebo.
"""

import math

import numpy as np

from il_control.observation import (laserscan_to_sectors, relative_goal,
                                    scan_to_points)
from il_control.policies import make_policy
from il_control.safety import ESTOP_FRONT, SLOWDOWN, SafetyConfig, SafetyFilter
from il_control.sim_world import SimWorld, integrate_unicycle

LIDAR_YAW = math.pi / 2
N = 360
INC = 2 * math.pi / N
ANGLES = -math.pi + np.arange(N) * INC


def rollout(policy_type, steps=600, dt=0.1, obstacles=(0.0, 0.0, 0.35, 1.5, 1.2, 0.3),
            safety_cfg=None):
    world = SimWorld(4.0, list(obstacles))
    policy = make_policy(policy_type)
    safety = SafetyFilter(safety_cfg or SafetyConfig())
    x, y, yaw = -3.0, 0.0, 0.0
    goal = (3.0, 0.0)
    reasons, t = [], 0.0
    for _ in range(steps):
        r = world.raycast(x, y, yaw + LIDAR_YAW, ANGLES, 12.0)
        d, a = relative_goal(x, y, yaw, *goal)
        if d < 0.25:
            return 'goal', reasons
        scan = laserscan_to_sectors(r, -math.pi, INC, 0.15, 12.0, 90, 12.0, LIDAR_YAW)
        v, w = policy.predict({'scan': scan, 'goal': [d, a]})
        pts = scan_to_points(r, -math.pi, INC, 0.15, 12.0, LIDAR_YAW)
        res = safety.filter(v, w, t, t, t, pts)
        reasons.append(res.reason)
        x2, y2, yaw2 = integrate_unicycle(x, y, yaw, res.v, res.w, dt)
        if world.in_collision(x2, y2):
            return 'collision', reasons
        x, y, yaw, t = x2, y2, yaw2, t + dt
    return 'timeout', reasons


def test_reactive_mock_reaches_goal_without_collision():
    outcome, _ = rollout('mock_reactive')
    assert outcome == 'goal'


def test_blind_policy_is_stopped_by_safety_layer():
    """mock_goal no esquiva: sin la capa de seguridad chocaría con el obstáculo."""
    outcome, reasons = rollout('mock_goal', steps=300)
    assert outcome == 'timeout'  # ni choca ni llega: se queda parado delante
    assert {SLOWDOWN, ESTOP_FRONT} & set(reasons)


def test_blind_policy_without_safety_would_collide():
    """Control negativo: sin la parada por proximidad, la misma política choca."""
    outcome, _ = rollout('mock_goal', steps=300,
                         safety_cfg=SafetyConfig(proximity_check=False))
    assert outcome == 'collision'
