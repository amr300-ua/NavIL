import numpy as np
import pytest
import torch

from il_policy.contract import INPUT_SIZE, ContractConfig
from il_policy.model import MLPPolicy, build_model
from il_policy.policy import ILPolicy, predict

CFG = ContractConfig()


def test_mlp_output_shape_and_bounds():
    model = MLPPolicy()
    y = model(torch.randn(16, INPUT_SIZE) * 10)
    assert y.shape == (16, 2)
    assert torch.all(y.abs() <= 1.0)


def test_build_model_from_config():
    m = MLPPolicy(hidden=(32, 16))
    m2 = build_model(m.config)
    m2.load_state_dict(m.state_dict())


def test_contract_predict_function():
    v, w = predict({'scan': [2.0] * 90, 'goal': [1.0, 0.2]})
    assert isinstance(v, float) and isinstance(w, float)
    assert abs(v) <= CFG.v_max and abs(w) <= CFG.w_max


def test_untrained_policy_handles_bad_lidar():
    policy = ILPolicy.untrained()
    v, w = policy.predict({'scan': [float('inf')] * 45 + [float('nan')] * 45,
                           'goal': np.array([1.0, 0.0])})
    assert np.isfinite(v) and np.isfinite(w)


def test_predict_batch_matches_predict():
    policy = ILPolicy.untrained()
    rng = np.random.default_rng(0)
    scan = rng.uniform(0.2, 5, (8, 90)).astype(np.float32)
    goal = np.stack([rng.uniform(0, 5, 8), rng.uniform(-3, 3, 8)], 1).astype(np.float32)
    vb, wb = policy.predict_batch(scan, goal)
    for i in range(8):
        v, w = policy.predict({'scan': scan[i], 'goal': goal[i]})
        assert v == pytest.approx(vb[i], abs=1e-5) and w == pytest.approx(wb[i], abs=1e-5)
