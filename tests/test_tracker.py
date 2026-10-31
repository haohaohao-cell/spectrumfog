"""航迹层：起批、确认、滑行、门限、离群点、速度估计。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrumfog import aar
from spectrumfog.engine.sim import Simulation
from spectrumfog.engine.tracker import Tracker, smoothing_noise
from spectrumfog.model.scenario import load_scenario
from spectrumfog.replay import Replay, build_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
R = np.eye(2) * 100.0  # 10 m 标准差


def _feed(tracker: Tracker, positions, t0: float = 0.0, dt: float = 2.0):
    events = []
    for index, position in enumerate(positions):
        events.extend(tracker.step([(position, R)], t0 + index * dt))
    return events


def test_confirms_after_three_hits():
    tracker = Tracker(dt=2.0)

    events = _feed(tracker, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])

    assert [event["status"] for event in events] == [
        "tentative",
        "tentative",
        "confirmed",
    ]
    assert tracker.tracks[0].hits == 3


def test_tentative_track_survives_two_misses_then_is_dropped():
    tracker = Tracker(dt=2.0, tentative_max_miss=2)
    tracker.step([((0.0, 0.0), R)], 0.0)

    for index in range(2):
        events = tracker.step([], 2.0 + index * 2.0)
        assert not any(event["status"] == "lost" for event in events)

    events = tracker.step([], 6.0)
    assert any(event["status"] == "lost" for event in events)
    assert tracker.lost_ids() == [1]


def test_confirmed_track_coasts_then_is_lost():
    tracker = Tracker(dt=2.0, max_coast=3)
    _feed(tracker, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    assert tracker.tracks[0].status == "confirmed"

    for step in range(3):
        events = tracker.step([], 6.0 + step * 2.0)
        assert not any(event["status"] == "lost" for event in events)

    events = tracker.step([], 12.0)
    assert any(event["status"] == "lost" for event in events)


def test_outlier_is_absorbed_without_corrupting_the_track():
    tracker = Tracker(dt=2.0)
    _feed(tracker, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    before = tracker.tracks[0].position

    # 5 km 的离群点：在关联半径内、在更新半径外 → 算漏检，不另起航迹
    tracker.step([((5000.0, 0.0), R)], 6.0)

    track = tracker.tracks[0]
    assert len(tracker.tracks) == 1
    assert track.misses == 1
    assert track.hits == 3  # 没有拿离群点更新
    # 位置只按"预测"推进（速度 × dt），没有被 5 km 的点拽走
    vx, vy = track.velocity
    assert track.position[0] == pytest.approx(before[0] + vx * 2.0, abs=1e-6)
    assert track.position[1] == pytest.approx(before[1] + vy * 2.0, abs=1e-6)


def test_far_fix_starts_a_second_track():
    tracker = Tracker(dt=2.0)
    _feed(tracker, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])

    tracker.step([((20000.0, 0.0), R)], 6.0)

    assert len({track.track_id for track in tracker.tracks}) == 2


def test_velocity_is_estimated_from_positions():
    tracker = Tracker(dt=2.0)
    # 100 m/s 向东，每帧走 200 m
    _feed(tracker, [(i * 200.0, 0.0) for i in range(8)])

    vx, vy = tracker.tracks[0].velocity
    assert vx == pytest.approx(100.0, abs=10.0)
    assert vy == pytest.approx(0.0, abs=1.0)


def test_process_noise_grows_with_dt():
    small = smoothing_noise(1.0, 20.0)
    large = smoothing_noise(2.0, 20.0)

    assert large[0, 0] > small[0, 0]
    assert small[0, 0] == pytest.approx(400.0 * 1.0 / 4.0)


def test_sim_emits_track_events_and_summary():
    result = Simulation(load_scenario(REPO_ROOT / "scenarios" / "01_duel.yaml")).run()
    tracks = [event for event in result.events if event["kind"] == "track"]

    assert tracks, "应该产生 track 事件"
    assert {"track_id", "status", "pos", "vel", "hits", "misses"} <= set(tracks[0])
    summary = result.summary
    assert 0.0 <= summary["track_confirmed_fraction"] <= 1.0
    assert summary["first_track_s"] is not None
    assert summary["n_tracks"] >= 1


def test_track_events_reach_the_replay_frames():
    result = Simulation(load_scenario(REPO_ROOT / "scenarios" / "01_duel.yaml")).run()
    frames = build_frames([e for e in result.events if e["kind"] != "meta"])

    assert any(frame.tracks for frame in frames)


def test_aar_reports_the_track_timeline():
    result = Simulation(load_scenario(REPO_ROOT / "scenarios" / "01_duel.yaml")).run()
    replay = Replay(
        meta=result.meta, frames=build_frames([e for e in result.events if e["kind"] != "meta"])
    )

    report = aar.analyse(replay, emitter="jammer_b")

    assert report.track_first_s is not None
    assert report.track_confirmed_fraction is not None
    assert "起批" in report.sentence
    assert any(event.kind == "航迹确认" for event in report.timeline)
