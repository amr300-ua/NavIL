"""Runner de episodios: escenarios, decisión de fin, métricas y escritura."""

import csv
import json
import math
import os

import pytest

from il_control import episodes as E
from il_control.episodes import EpisodeLimits, EpisodeTracker, ResultsWriter, summarize

HERE = os.path.dirname(__file__)
SCENARIOS = os.path.join(HERE, '..', 'config', 'scenarios.yaml')


def test_load_scenarios_navil_expert_format(tmp_path):
    sc = E.load_scenarios(SCENARIOS)
    assert sc['ruta_validada'] == [(-3.81, -0.725, 0.0), (-1.372, 2.218, 0.0)]
    # Formato largo de Lucas (listas de dicts en varias líneas)
    f = tmp_path / 's.yaml'
    f.write_text('escenarios:\n  a:\n    waypoints:\n      - x: 1\n        y: 2\n')
    assert E.load_scenarios(str(f)) == {'a': [(1.0, 2.0, 0.0)]}
    f.write_text('escenarios:\n  vacio:\n    waypoints: []\n')
    with pytest.raises(ValueError):
        E.load_scenarios(str(f))


def test_waypoints_from_flat():
    assert E.waypoints_from_flat([1, 2, 90, 3, 4, 0]) == [(1, 2, 90), (3, 4, 0)]
    with pytest.raises(ValueError):
        E.waypoints_from_flat([1, 2])


def track(**kw):
    return EpisodeTracker((5.0, 0.0), 0.0, (0.0, 0.0), EpisodeLimits(**kw))


def test_success_by_distance():
    t = track()
    t.update_map(1.0, 4.9, 0.0)
    assert t.check(1.0) == E.SUCCESS
    assert t.check(1.0, use_distance_success=False) is None  # modo Nav2


def test_timeout_scales_with_distance():
    t = track(timeout_base=10, timeout_per_meter=2, stuck_window=1e9)
    assert t.timeout == pytest.approx(20.0)
    t.update_map(19.0, 1.0, 0.0)
    assert t.check(19.0) is None
    t.update_map(21.0, 1.0, 0.0)
    assert t.check(21.0) == E.TIMEOUT


def test_stuck_detection():
    t = track(stuck_window=5.0, stuck_distance=0.1)
    for k in range(60):
        t.update_map(k * 0.1, 1.0 + 0.001 * k, 0.0)
    assert t.check(5.9) == E.STUCK
    t = track(stuck_window=5.0, stuck_distance=0.1)
    for k in range(60):
        t.update_map(k * 0.1, 1.0 + 0.05 * k, 0.0)
    assert t.check(5.9) is None


def test_collision_ends_episode_unless_disabled():
    t = track()
    t.update_map(1.0, 1.0, 0.0)
    t.add_collision()
    assert t.check(1.0) == E.COLLISION
    assert t.check(1.0, stop_on_collision=False) is None


def test_path_length_uses_odom_and_ignores_jumps():
    t = track()
    for x in [0.0, 0.5, 1.0, 1.5]:
        t.update_odom(x, 0.0)
    t.update_odom(10.0, 0.0)  # salto (reinicio de odometría)
    assert t.path_length == pytest.approx(1.5)


def make_result(outcome, time_s=10.0, length=5.0, coll=0):
    t = track()
    for k in range(int(length * 10) + 1):
        t.update_odom(k * 0.1, 0)
    t.update_map(time_s, 5.0 if outcome == E.SUCCESS else 2.0, 0.0)
    for ms in [1.0, 2.0, 3.0]:
        t.add_latency(ms)
    t.add_scan_min(0.4)
    for _ in range(coll):
        t.add_collision()
    return t.result(outcome, episode_id='x', run_id='r', scenario='s', controller='policy',
                    policy='p', repetition=1, waypoint_index=1, waypoint_total=1)


def test_result_metrics():
    r = make_result(E.SUCCESS, 10.0, 5.0)
    assert r.success and r.path_efficiency == pytest.approx(1.0)
    assert r.mean_v == pytest.approx(0.5) and r.inference_ms_mean == pytest.approx(2.0)
    assert r.inference_hz == pytest.approx(500.0)


def test_summary_and_writer(tmp_path):
    rs = [make_result(E.SUCCESS, 10), make_result(E.SUCCESS, 20),
          make_result(E.COLLISION, coll=1), make_result(E.TIMEOUT)]
    s = summarize(rs)
    assert s['success_rate'] == 0.5 and s['collisions_total'] == 1
    assert s['time_s_mean_success'] == pytest.approx(15.0)
    assert s['outcomes'][E.TIMEOUT] == 1
    w = ResultsWriter(str(tmp_path / 'run'), {'controller': 'policy'})
    for r in rs:
        w.add(r)
    w.write_summary()
    rows = list(csv.DictReader((tmp_path / 'run' / 'episodes.csv').open()))
    assert len(rows) == 4 and rows[0]['outcome'] == 'success'
    data = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert data['summary']['episodes'] == 4 and data['meta']['controller'] == 'policy'


def test_nan_metrics_written_as_empty(tmp_path):
    t = track()
    r = t.result(E.TIMEOUT, episode_id='x', run_id='r', scenario='s', controller='policy',
                 policy='p', repetition=1, waypoint_index=1, waypoint_total=1)
    row = r.to_row()
    assert row['inference_ms_mean'] == '' and row['path_efficiency'] == ''
    assert not math.isnan(summarize([r])['success_rate'])
