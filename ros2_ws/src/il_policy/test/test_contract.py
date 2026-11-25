import math

import numpy as np
import pytest

from il_policy.contract import (INPUT_SIZE, ContractConfig, build_features,
                                denormalize_targets, normalize_targets,
                                observation_to_features, scan_angles, validate_dataset)

CFG = ContractConfig()


def test_feature_vector_shape_and_range():
    x = observation_to_features({'scan': np.full(90, 3.0), 'goal': [2.0, 0.5]}, CFG)
    assert x.shape == (INPUT_SIZE,)
    assert x.dtype == np.float32
    assert np.all(x[:90] >= 0) and np.all(x[:90] <= 1)


def test_inf_nan_are_sanitized():
    scan = [float('inf')] * 30 + [float('nan')] * 30 + [-1.0] * 30
    x = build_features(scan, [1.0, 0.0], CFG)
    assert np.all(np.isfinite(x))
    assert np.allclose(x[:60], 1.0)   # sin retorno = libre (rango máximo)
    assert np.allclose(x[60:90], 0.0)


def test_angle_is_wrapped():
    x = build_features(np.ones(90), [1.0, 3 * math.pi / 2], CFG)
    assert x[-1] == pytest.approx(-0.5, abs=1e-5)


def test_wrong_sizes_raise():
    with pytest.raises(ValueError):
        build_features(np.ones(89), [1.0, 0.0], CFG)
    with pytest.raises(ValueError):
        build_features(np.ones(90), [1.0], CFG)
    with pytest.raises(KeyError):
        observation_to_features({'scan': np.ones(90)}, CFG)


def test_targets_roundtrip():
    t = normalize_targets(np.array([0.15]), np.array([-0.75]), CFG)
    v, w = denormalize_targets(t, CFG)
    assert v[0] == pytest.approx(0.15) and w[0] == pytest.approx(-0.75)


def test_scan_angle_convention():
    a = scan_angles()
    assert a[0] == 0.0                                  # frente
    assert a[45] == pytest.approx(-math.pi)             # detrás
    assert a[22] == pytest.approx(math.radians(88))     # ~izquierda


def test_validate_dataset_detects_bad_shapes():
    n = 5
    ok = {'scan': np.zeros((n, 90)), 'goal': np.zeros((n, 2)), 'v': np.zeros(n),
          'w': np.zeros(n), 'episode_id': np.zeros(n)}
    assert validate_dataset(ok, CFG) == n
    with pytest.raises(ValueError):
        validate_dataset({**ok, 'scan': np.zeros((n, 80))}, CFG)
    with pytest.raises(KeyError):
        validate_dataset({k: v for k, v in ok.items() if k != 'w'}, CFG)
