"""IMM（双模型）+ JPDA（概率数据关联）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrumfog.engine.imm import IMMTracker
from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
R = np.eye(2) * 400.0  # 20 m 标准差


def _straight_then_turn(tracker: IMMTracker, *, frames: int = 40, seed: int = 0):
    """直线 20 帧后 90° 急转；返回跟踪器与每帧的机动模型概率。"""
    rng = np.random.default_rng(seed)
    x, y, vx, vy = 0.0, 0.0, 100.0, 0.0
    maneuver_probability = []
    for k in range(frames):
        if k == 20:
            vx, vy = 0.0, 100.0  # 90° 急转
        x += vx
        y += vy
        tracker.step([(np.array([x, y]) + rng.normal(0, 20, 2), R)], float(k))
        maneuver_probability.append(float(tracker.tracks[0].mode_probabilities[1]))
    return tracker, maneuver_probability


def test_imm_tracks_a_constant_velocity_target():
    tracker = IMMTracker(dt=1.0)
    for k in range(15):
        tracker.step([(np.array([100.0 * k, 0.0]), R)], float(k))

    vx, vy = tracker.tracks[0].velocity
    assert vx == pytest.approx(100.0, abs=10.0)
    assert vy == pytest.approx(0.0, abs=5.0)


def test_imm_switches_to_the_maneuver_model_on_a_turn():
    tracker, maneuver = _straight_then_turn(IMMTracker(dt=1.0))

    # 转弯前：平稳模型占优；转弯后几帧：机动模型翻上来
    assert maneuver[15] < 0.1
    assert max(maneuver[21:32]) > 0.5
    # 转弯结束后又回到平稳模型（IMM 会自己收回来）
    assert tracker.tracks[0].mode_probabilities[1] < 0.1


def test_mode_probabilities_stay_normalised():
    tracker = IMMTracker(dt=1.0)
    for k in range(25):
        tracker.step([(np.array([50.0 * k, 10.0 * k]), R)], float(k))
        assert tracker.tracks[0].mode_probabilities.sum() == pytest.approx(1.0)


def test_mode_floor_keeps_both_models_alive():
    """概率一旦到 0 就再也回不来——这是 IMM 的经典坑，用下限防住。"""
    _, maneuver = _straight_then_turn(IMMTracker(dt=1.0))

    assert all(probability > 0.0 for probability in maneuver)


def test_pda_combines_two_close_measurements():
    tracker = IMMTracker(dt=1.0)
    tracker.step([(np.array([0.0, 0.0]), R)], 0.0)
    before = tracker.tracks[0].position

    # 两条相距 200 m 的量测同时在门内
    tracker.step(
        [
            (np.array([100.0, 0.0]), R),
            (np.array([300.0, 0.0]), R),
        ],
        1.0,
    )

    assert len(tracker.tracks) == 1
    after = tracker.tracks[0].position[0]
    assert after > before[0]  # 被拉向两条量测之间
    assert 0.0 < after < 300.0


def test_track_is_lost_after_coasting():
    tracker = IMMTracker(dt=1.0, max_coast=2)
    for k in range(4):
        tracker.step([(np.array([10.0 * k, 0.0]), R)], float(k))

    lost = []
    for k in range(4, 9):
        lost.extend(
            event for event in tracker.step([], float(k)) if event["status"] == "lost"
        )

    assert lost


def test_far_measurement_starts_a_second_track():
    tracker = IMMTracker(dt=1.0)
    for k in range(4):
        tracker.step([(np.array([10.0 * k, 0.0]), R)], float(k))

    tracker.step([(np.array([40000.0, 0.0]), R)], 5.0)

    assert len({track.track_id for track in tracker.tracks}) == 2


def test_sim_can_use_the_imm_tracker():
    result = Simulation(
        load_scenario(REPO_ROOT / "scenarios" / "04_fog.yaml"), tracker="imm"
    ).run()
    tracks = [
        event
        for event in result.events
        if event["kind"] == "track" and event.get("source") != "radar"
    ]

    assert tracks
    assert all(event.get("filter") == "imm" for event in tracks)
    assert result.summary["track_confirmed_fraction"] is not None


def test_invalid_tracker_name_is_rejected():
    scenario = load_scenario(REPO_ROOT / "scenarios" / "04_fog.yaml")
    with pytest.raises(ValueError, match="tracker"):
        Simulation(scenario, tracker="ekf")
