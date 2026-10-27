"""回放格式：一行一事件、可独立解析、字节稳定。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectrumfog.replay import (
    EVENT_KINDS,
    Replay,
    build_frames,
    iter_events,
    read_replay,
    write_replay,
)


def _events() -> list[dict]:
    return [
        {"kind": "meta", "t": 0.0, "scenario": "demo", "seed": 1, "dt": 1.0,
         "duration": 2.0, "pd_threshold": 0.5, "unjammed_detection_range_m": 1000.0,
         "radar": "r", "jammer": "j", "target": "t"},
        {"kind": "state", "t": 0.0, "platform": "r", "pos": [0.0, 0.0], "emitter": "radar", "on": True},
        {"kind": "state", "t": 0.0, "platform": "j", "pos": [500.0, 0.0], "emitter": "noise_jammer", "on": True},
        {"kind": "link", "t": 0.0, "radar": "r", "jammer": "j", "target": "t",
         "range_target_m": 300.0, "range_jammer_m": 500.0, "js": 2.5, "pd": 0.1,
         "burn_through_m": 120.0, "detection_ring_m": 80.0},
        {"kind": "detection", "t": 0.0, "sensor": "r", "target": "t", "pd": 0.1,
         "draw": 0.42, "detected": False},
        {"kind": "link", "t": 1.0, "radar": "r", "jammer": "j", "target": "t",
         "range_target_m": 300.0, "range_jammer_m": 400.0, "js": 3.9, "pd": 0.05,
         "burn_through_m": 100.0, "detection_ring_m": 60.0},
        {"kind": "detection", "t": 1.0, "sensor": "r", "target": "t", "pd": 0.05,
         "draw": 0.91, "detected": False},
    ]


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "demo.jsonl"
    written = write_replay(path, _events())
    assert written == len(_events())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_events())
    # 每一行都必须能独立解析成对象（不依赖上下文）
    for line in lines:
        payload = json.loads(line)
        assert set(payload) >= {"kind", "t"}
        assert payload["kind"] in EVENT_KINDS

    assert read_replay(path) == _events()


def test_write_is_byte_stable(tmp_path):
    """同样的输入 → 同样的字节，方便直接比对/哈希。"""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_replay(first, _events())
    write_replay(second, _events())
    assert first.read_bytes() == second.read_bytes()


def test_bad_line_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"kind": "state", "t": 0, "platform": "r", "pos": [0, 0]}\n'
        "this is not json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2"):
        list(iter_events(path))


def test_unknown_event_kind_rejected(tmp_path):
    path = tmp_path / "kind.jsonl"
    path.write_text('{"kind": "teleport", "t": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="未知事件类型"):
        list(iter_events(path))


def test_replay_requires_meta(tmp_path):
    path = tmp_path / "nometa.jsonl"
    write_replay(path, [event for event in _events() if event["kind"] != "meta"])
    with pytest.raises(ValueError, match="meta"):
        Replay.load(path)


def test_frames_group_events_by_time(tmp_path):
    path = tmp_path / "demo.jsonl"
    write_replay(path, _events())
    replay = Replay.load(path)

    assert replay.meta["scenario"] == "demo"
    assert replay.times == (0.0, 1.0)
    assert replay.duration == 1.0

    first, second = replay.frames
    assert first.positions == {"r": (0.0, 0.0), "j": (500.0, 0.0)}
    assert first.emitters == {"r": "radar", "j": "noise_jammer"}
    assert first.link["js"] == pytest.approx(2.5)
    assert first.detection["detected"] is False
    assert second.link["js"] == pytest.approx(3.9)


def test_build_frames_does_not_depend_on_event_order():
    """日志被按时间重排后，帧内容必须一致（只依赖 t）。"""
    events = [event for event in _events() if event["kind"] != "meta"]
    ordered = build_frames(events)
    shuffled = build_frames(list(reversed(events)))
    assert ordered == shuffled
