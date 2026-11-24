"""Backends de política e integración opcional con il_policy."""

import math

import numpy as np
import pytest

from il_control.policies import (MockGoalPolicy, MockReactivePolicy, ZeroPolicy,
                                 make_policy)

FREE = np.full(90, 5.0)


def obs(goal, scan=FREE):
    return {'scan': scan, 'goal': goal}


def test_factory():
    assert isinstance(make_policy('zero'), ZeroPolicy)
    assert isinstance(make_policy('mock_goal'), MockGoalPolicy)
    assert isinstance(make_policy('mock_reactive'), MockReactivePolicy)
    with pytest.raises(ValueError):
        make_policy('transformer')


def test_goal_steering_sign():
    p = MockGoalPolicy()
    v, w = p.predict(obs([2.0, 0.0]))
    assert v > 0.2 and w == 0.0
    assert p.predict(obs([2.0, 1.0]))[1] > 0      # objetivo a la izquierda -> gira izq.
    assert p.predict(obs([2.0, -1.0]))[1] < 0
    assert p.predict(obs([2.0, math.pi]))[0] == 0  # detrás: gira sin avanzar


def test_reactive_avoids_and_slows():
    scan = FREE.copy()
    scan[[0, 1, 89]] = 0.4    # obstáculo delante
    scan[60:80] = 0.8         # derecha más ocupada
    v0, _ = MockReactivePolicy().predict(obs([3.0, 0.0]))
    v, w = MockReactivePolicy().predict(obs([3.0, 0.0], scan))
    assert v < v0 and w > 0   # frena y gira hacia la izquierda (más libre)


@pytest.mark.parametrize('ptype', ['zero', 'mock_goal', 'mock_reactive'])
def test_outputs_finite_and_bounded(ptype):
    p = make_policy(ptype)
    rng = np.random.default_rng(1)
    for _ in range(200):
        v, w = p.predict(obs([rng.uniform(0, 10), rng.uniform(-math.pi, math.pi)],
                             rng.uniform(0.1, 12, 90)))
        assert math.isfinite(v) and math.isfinite(w)
        assert -0.3 <= v <= 0.3 and abs(w) <= 1.5


def test_il_policy_backend_untrained():
    """Solo si il_policy y PyTorch están en el entorno (rama feature/il-policy)."""
    pytest.importorskip('torch')
    pytest.importorskip('il_policy.policy')
    p = make_policy('il_policy')
    assert p.scan_size == 90
    v, w = p.predict(obs([2.0, 0.3]))
    assert abs(v) <= p.v_max and abs(w) <= p.w_max
