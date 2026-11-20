"""Prueba de extremo a extremo con datos ficticios (objetivo del ciclo 1)."""

import json

import numpy as np
import pytest

from il_policy.contract import ContractConfig
from il_policy.dataset import load_npz, split_by_episode
from il_policy.policy import ILPolicy
from il_policy.synthetic import generate_dataset
from il_policy.train import DEFAULTS, train


@pytest.fixture(scope='module')
def dummy_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp('data') / 'dummy.npz'
    data = generate_dataset(n_episodes=100, seed=1, max_steps=400)
    np.savez_compressed(path, **data)
    return path


def test_dummy_dataset_follows_contract(dummy_npz):
    data = load_npz(dummy_npz, ContractConfig())
    assert data['scan'].shape[1] == 90
    assert len(np.unique(data['episode_id'])) >= 80


def test_split_does_not_mix_episodes():
    ep = np.repeat(np.arange(20), 10)
    tr, va, te = split_by_episode(ep, 0.2, 0.2, seed=0)
    s_tr, s_va, s_te = set(ep[tr]), set(ep[va]), set(ep[te])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    assert len(tr) + len(va) + len(te) == len(ep)


def test_training_learns_better_than_baseline(dummy_npz, tmp_path):
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg['training'].update(epochs=25, patience=25, batch_size=128)
    metrics = train([dummy_npz], tmp_path / 'model', cfg, verbose=False)

    test = metrics['test']
    assert test['mae_v'] < 0.7 * test['baseline_mae_v']
    assert test['mae_w'] < test['baseline_mae_w']
    for f in ('best.pt', 'metrics.json', 'history.csv'):
        assert (tmp_path / 'model' / f).exists()

    policy = ILPolicy.load(str(tmp_path / 'model' / 'best.pt'))
    v_free, _ = policy.predict({'scan': [6.0] * 90, 'goal': [3.0, 0.0]})
    _, w_left = policy.predict({'scan': [6.0] * 90, 'goal': [2.0, 1.2]})
    _, w_right = policy.predict({'scan': [6.0] * 90, 'goal': [2.0, -1.2]})
    assert v_free > 0.1          # camino libre y objetivo delante -> avanza
    assert w_left > 0 > w_right  # gira hacia el objetivo
