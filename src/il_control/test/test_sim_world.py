"""Simulador stub: raycasting e integración cinemática."""

import math

import numpy as np
import pytest

from il_control.observation import laserscan_to_sectors
from il_control.sim_world import SimWorld, integrate_unicycle


def test_raycast_walls_and_obstacle():
    w = SimWorld(half=4.0, obstacles=[2.0, 0.0, 0.5])
    r = w.raycast(0.0, 0.0, 0.0, np.array([0.0, math.pi / 2, math.pi]), 12.0)
    assert r == pytest.approx([1.5, 4.0, 4.0])


def test_collision():
    w = SimWorld(obstacles=[1.0, 0.0, 0.3], robot_radius=0.18)
    assert w.in_collision(0.55, 0.0) and not w.in_collision(0.0, 0.0)
    assert w.in_collision(3.9, 0.0)


def test_unicycle():
    x, y, yaw = integrate_unicycle(0, 0, 0, 1.0, 0.0, 1.0)
    assert (x, y, yaw) == pytest.approx((1, 0, 0))
    x, y, yaw = integrate_unicycle(0, 0, 0, math.pi / 2, math.pi / 2, 1.0)  # cuarto de círculo r=1
    assert (x, y, yaw) == pytest.approx((1, 1, math.pi / 2))


def test_rotated_lidar_end_to_end():
    """fake_sim con LiDAR girado + preprocesado con yaw de TF -> obstáculo en scan[0]."""
    w = SimWorld(obstacles=[1.0, 0.0, 0.2])
    lidar_yaw = math.pi / 2
    angles = -math.pi + np.arange(360) * (2 * math.pi / 360)
    r = w.raycast(0.0, 0.0, 0.0 + lidar_yaw, angles, 12.0)
    s = laserscan_to_sectors(r, -math.pi, 2 * math.pi / 360, 0.15, 12.0, 90, 12.0, lidar_yaw)
    assert s[0] == pytest.approx(0.8, abs=0.01) and s[45] == pytest.approx(4.0, abs=0.01)
