"""Capa de seguridad: límites, watchdogs, parada por proximidad y rampa."""

import math

import numpy as np
import pytest

from il_control import safety as S
from il_control.safety import SafetyConfig, SafetyFilter, corridor_clearance

FREE = np.empty((0, 2))


def make(**kw):
    kw.setdefault('max_lin_accel', 0.0)  # sin rampa salvo en su test
    kw.setdefault('max_ang_accel', 0.0)
    return SafetyFilter(SafetyConfig(**kw))


def run(f, v, w, pts=FREE, now=10.0, cmd_age=0.0, scan_age=0.0):
    return f.filter(v, w, now, now - cmd_age, now - scan_age, pts)


def test_passthrough_and_clamp():
    f = make()
    r = run(f, 0.2, 0.5)
    assert (r.v, r.w, r.reason) == (0.2, 0.5, S.OK)
    r = run(f, 1.0, -9.0)
    assert (r.v, r.w, r.reason) == (0.30, -1.50, S.CLAMPED)
    assert run(f, -1.0, 0.0).v == -0.10


@pytest.mark.parametrize('v, w', [(math.nan, 0.0), (0.1, math.inf)])
def test_invalid_command_stops(v, w):
    r = run(make(), v, w)
    assert (r.v, r.w, r.reason) == (0.0, 0.0, S.INVALID_CMD)


def test_scan_watchdog():
    r = run(make(), 0.2, 0.3, scan_age=0.6)
    assert (r.v, r.w, r.reason) == (0.0, 0.0, S.NO_SCAN)
    f = make()
    assert f.filter(0.2, 0.3, 10.0, 10.0, None, FREE).reason == S.NO_SCAN


def test_cmd_watchdog():
    r = run(make(), 0.2, 0.3, cmd_age=0.6)
    assert (r.v, r.w, r.reason) == (0.0, 0.0, S.NO_CMD)


def test_estop_front_keeps_rotation():
    pts = np.array([[0.18 + 0.10, 0.0]])  # 10 cm libres delante del casco
    r = run(make(), 0.25, 0.8, pts)
    assert r.reason == S.ESTOP_FRONT and r.v == 0.0 and r.w == 0.8
    assert r.clearance == pytest.approx(0.10)


def test_slowdown_is_linear():
    c = SafetyConfig()
    mid = (c.stop_distance + c.slow_distance) / 2
    r = run(make(), 0.30, 0.0, np.array([[c.robot_radius + mid, 0.0]]))
    assert r.reason == S.SLOWDOWN and r.v == pytest.approx(0.15)


def test_obstacle_outside_corridor_is_ignored():
    pts = np.array([[0.3, 0.40], [0.3, -0.40], [-0.25, 0.0]])  # a los lados y detrás
    assert run(make(), 0.25, 0.0, pts).reason == S.OK


def test_reverse_checks_rear():
    pts = np.array([[-0.30, 0.0]])
    assert run(make(), -0.1, 0.0, pts).reason == S.ESTOP_REAR
    assert run(make(), 0.2, 0.0, pts).reason == S.OK  # delante libre


def test_proximity_can_be_disabled_per_call():
    f = make()
    pts = np.array([[0.25, 0.0]])
    assert f.filter(0.2, 0, 1.0, 1.0, 1.0, pts, proximity_check=False).v == 0.2


def test_ramp_limits_acceleration_but_not_estops():
    f = make(max_lin_accel=1.0, max_ang_accel=2.0)
    run(f, 0.0, 0.0, now=0.0)
    r = run(f, 0.3, 1.5, now=0.05)  # dt = 50 ms
    assert r.v == pytest.approx(0.05) and r.w == pytest.approx(0.10)
    r = run(f, 0.3, 0.0, np.array([[0.2, 0.0]]), now=0.10)
    assert r.v == 0.0 and r.reason == S.ESTOP_FRONT  # parada inmediata


def test_corridor_clearance():
    assert corridor_clearance(None, 0.2, 0.18) == math.inf
    pts = np.array([[1.0, 0.1], [0.5, 0.3]])
    assert corridor_clearance(pts, 0.2, 0.18) == pytest.approx(0.82)


def test_invalid_config():
    with pytest.raises(ValueError):
        SafetyFilter(SafetyConfig(stop_distance=0.6, slow_distance=0.5))
