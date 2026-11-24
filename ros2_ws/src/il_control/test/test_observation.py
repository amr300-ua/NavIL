"""Preprocesado LiDAR/objetivo: convención idéntica al contrato de il_policy."""

import math

import numpy as np
import pytest

from il_control.observation import (clean_ranges, goal_in_base_frame,
                                    laserscan_to_sectors, relative_goal,
                                    scan_to_points, transform_point_2d,
                                    yaw_from_quaternion)

N_RAYS = 360
INC = 2 * math.pi / N_RAYS
ANGLE_MIN = -math.pi  # convención típica de un LiDAR 360°: empieza detrás


def scan_with_obstacle(at_deg: float, dist: float = 0.5, free: float = math.inf,
                       lidar_yaw_deg: float = 0.0):
    """LaserScan sintético con un obstáculo en ``at_deg`` (marco base_link)."""
    angles = ANGLE_MIN + np.arange(N_RAYS) * INC + math.radians(lidar_yaw_deg)
    diff = np.abs((angles - math.radians(at_deg) + math.pi) % (2 * math.pi) - math.pi)
    r = np.full(N_RAYS, free)
    r[diff < math.radians(1.5)] = dist
    return r


def to90(r, yaw=0.0):
    return laserscan_to_sectors(r, ANGLE_MIN, INC, 0.15, 12.0, 90, 12.0, yaw)


@pytest.mark.parametrize('deg, index', [(0, 0), (90, 22.5), (180, 45), (-90, 67.5),
                                        (45, 11.25)])
def test_sector_convention_matches_contract(deg, index):
    """scan[0] delante, antihorario, 4° por sector (il_policy.contract)."""
    s = to90(scan_with_obstacle(deg))
    hit = np.flatnonzero(s < 1.0)
    assert hit.size >= 1
    assert abs(hit.mean() - index) <= 1.0


def test_rotated_lidar_uses_yaw_offset():
    """Con el LiDAR girado 90°, el obstáculo frontal debe seguir en scan[0]."""
    r = scan_with_obstacle(0, lidar_yaw_deg=90)
    assert to90(r, math.radians(90))[0] < 1.0
    assert to90(r, 0.0)[0] > 1.0  # sin compensar quedaría en otro sector


def test_min_per_sector_keeps_thin_obstacles():
    r = np.full(N_RAYS, 5.0)
    r[180] = 0.4  # un solo rayo (pata de mesa) justo delante
    assert to90(r)[0] == pytest.approx(0.4)


def test_special_values_rep117():
    r = clean_ranges([math.inf, -math.inf, math.nan, 0.05, 20.0, 3.0], 0.15, 12.0, 12.0)
    assert r[0] == 12.0 and r[1] == 0.0 and r[4] == 12.0 and r[5] == 3.0
    assert math.isnan(r[2]) and math.isnan(r[3])  # se ignoran


def test_output_is_finite_and_bounded():
    r = np.random.default_rng(0).uniform(-1, 20, N_RAYS)
    r[::7] = math.nan
    r[::11] = math.inf
    s = to90(r)
    assert s.shape == (90,) and s.dtype == np.float32
    assert np.all(np.isfinite(s)) and np.all((s >= 0) & (s <= 12.0))


def test_partial_fov_fills_free():
    r = np.full(90, 2.0)  # solo 90 rayos de 1° delante (-45°..45°)
    s = laserscan_to_sectors(r, -math.pi / 4, math.radians(1), 0.1, 12, 90, 12.0)
    assert s[0] == pytest.approx(2.0) and s[45] == 12.0


def test_compatible_with_il_policy_contract():
    """Si il_policy está disponible, su contrato acepta nuestra salida sin cambios."""
    contract = pytest.importorskip('il_policy.contract')
    cfg = contract.ContractConfig()
    s = to90(scan_with_obstacle(30, 0.6))
    np.testing.assert_allclose(contract.sanitize_scan(s, cfg), s)
    assert contract.build_features(s, [2.0, 0.3], cfg).shape == (92,)
    # Misma convención de ángulos por índice
    np.testing.assert_allclose(contract.scan_angles(90)[[0, 22, 45]],
                               [0.0, math.radians(88), -math.pi], atol=1e-6)


def test_relative_goal_matches_il_data_example():
    """Mismo caso que el test de il_data: robot mirando a +y, objetivo (0, 2)."""
    yaw = yaw_from_quaternion(0, 0, math.sqrt(0.5), math.sqrt(0.5))
    d, a = relative_goal(0, 0, yaw, 0, 2)
    assert d == pytest.approx(2.0) and a == pytest.approx(0.0, abs=1e-9)
    d, a = relative_goal(0, 0, 0.0, 0, 2)
    assert a == pytest.approx(math.pi / 2)  # positivo = izquierda


def test_tf_path_equals_pose_path():
    """Transformar el objetivo con la TF base<-map da lo mismo que la fórmula de il_data."""
    rx, ry, ryaw, gx, gy = 1.0, -2.0, 0.7, 3.5, 1.0
    # TF base_link <- map = inversa de la pose del robot
    c, s = math.cos(-ryaw), math.sin(-ryaw)
    tx, ty = -(c * rx - s * ry), -(s * rx + c * ry)
    via_tf = goal_in_base_frame(*transform_point_2d(tx, ty, -ryaw, gx, gy))
    via_pose = relative_goal(rx, ry, ryaw, gx, gy)
    assert via_tf == pytest.approx(via_pose)


def test_scan_to_points_frame():
    r = scan_with_obstacle(0, 1.0)
    pts = scan_to_points(r, ANGLE_MIN, INC, 0.15, 12.0)
    assert len(pts) >= 1 and np.allclose(pts[:, 0], 1.0, atol=0.01)
    pts = scan_to_points(scan_with_obstacle(0, 1.0, lidar_yaw_deg=90), ANGLE_MIN, INC,
                         0.15, 12.0, yaw_offset=math.radians(90))
    assert np.allclose(pts[:, 0], 1.0, atol=0.01) and np.allclose(pts[:, 1], 0, atol=0.03)


# ---------------------------------------------------------------- modo interp
from il_control.observation import (laserscan_to_contract_interp,  # noqa: E402
                                    preprocess_scan)


def test_interp_matches_il_data():
    """Debe dar exactamente lo mismo que remuestrear_lidar de il_data."""
    du = pytest.importorskip('il_data.dataset_utils')
    rng = np.random.default_rng(7)
    cases = [
        (rng.uniform(0.1, 15, 360), -math.pi, INC, 0.0),
        (rng.uniform(0.1, 15, 640), -math.pi, 2 * math.pi / 640, 0.3),
        (rng.uniform(0.1, 15, 1440), 0.0, 2 * math.pi / 1440, math.pi / 2),
        (rng.uniform(0.1, 15, 270), -math.radians(135), math.radians(1), 0.0),  # FOV parcial
        (rng.uniform(0.1, 15, 360)[::-1], math.pi, -INC, 0.0),  # sentido horario
    ]
    for r, amin, inc, yaw in cases:
        r = r.copy()
        r[::13], r[::17], r[::19] = math.nan, math.inf, -math.inf
        ref = du.remuestrear_lidar(list(r), amin + yaw, inc, 12.0)
        mine = laserscan_to_contract_interp(r, amin, inc, 90, 12.0, yaw)
        np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-5)


def test_interp_convention_and_rotation():
    s = preprocess_scan(scan_with_obstacle(0, 0.5, free=5.0), ANGLE_MIN, INC, 0.15, 12.0)
    assert s[0] == pytest.approx(0.5) and s[45] == pytest.approx(5.0)
    r = scan_with_obstacle(0, 0.5, free=5.0, lidar_yaw_deg=90)
    s = preprocess_scan(r, ANGLE_MIN, INC, 0.15, 12.0, yaw_offset=math.radians(90))
    assert s[0] == pytest.approx(0.5)


def test_interp_can_miss_thin_obstacles_but_min_does_not():
    """Documenta la diferencia: un obstáculo de ~3° entre dos ángulos de muestreo
    desaparece con 'interp' (muestreo puntual) y se conserva con 'min'."""
    r = scan_with_obstacle(90, 0.5, free=5.0)  # rayos 89°-91°, muestras en 88° y 92°
    interp = preprocess_scan(r, ANGLE_MIN, INC, 0.15, 12.0, reduction='interp')
    mins = preprocess_scan(r, ANGLE_MIN, INC, 0.15, 12.0, reduction='min')
    assert interp.min() == pytest.approx(5.0) and mins.min() == pytest.approx(0.5)


def test_preprocess_dispatch():
    r = scan_with_obstacle(0, 0.5, free=5.0)
    assert preprocess_scan(r, ANGLE_MIN, INC, 0.15, 12.0, reduction='min')[0] < 0.6
    with pytest.raises(ValueError):
        preprocess_scan(r, ANGLE_MIN, INC, 0.15, 12.0, reduction='mean')
