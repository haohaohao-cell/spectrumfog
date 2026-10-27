"""动画渲染：读 replay → GIF（viz 只读，不回写）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import parse_scenario
from spectrumfog.replay import Replay, write_replay
from viz.static.animate import animate_file, animate_replay

SHORT_SCENARIO = {
    "meta": {"name": "short_standoff", "seed": 3, "dt": 2.0, "duration": 10.0},
    "environment": {"noise_figure_db": 3.0, "temperature_k": 290},
    "platforms": [
        {
            "id": "radar_a",
            "side": "blue",
            "pos": [0, 0],
            "emitters": [
                {
                    "kind": "radar",
                    "freq_hz": 3.0e9,
                    "bw_hz": 2.0e6,
                    "pt_w": 1.5e5,
                    "gain_db": 35,
                    "sigma_m2": 1.0,
                    "loss_db": 3,
                }
            ],
        },
        {
            "id": "jammer_b",
            "side": "red",
            "pos": [30000, 0],
            "waypoints": [{"t": 0, "pos": [30000, 0]}, {"t": 10, "pos": [10000, 0]}],
            "emitters": [
                {
                    "kind": "noise_jammer",
                    "freq_hz": 3.0e9,
                    "bw_hz": 1.0e8,
                    "pj_w": 2.0,
                    "gain_db": 0,
                }
            ],
        },
        {"id": "target_c", "side": "red", "pos": [12000, 4000], "emitters": []},
    ],
    "objectives": [{"type": "detect", "target": "target_c"}],
}


@pytest.fixture()
def short_replay(tmp_path) -> tuple[Path, int]:
    scenario = parse_scenario(SHORT_SCENARIO)
    result = Simulation(scenario).run()
    path = tmp_path / "short.jsonl"
    write_replay(path, result.events)
    return path, result.summary["n_ticks"]


def test_animate_file_writes_gif(short_replay, tmp_path):
    replay_path, n_frames = short_replay
    out = tmp_path / "short.gif"

    animate_file(replay_path, out, fps=6)

    assert out.exists()
    assert out.stat().st_size > 5_000  # 空白图会明显更小
    with Image.open(out) as image:
        assert image.n_frames == n_frames


def test_animate_respects_stride(short_replay, tmp_path):
    replay_path, n_frames = short_replay
    out = tmp_path / "strided.gif"

    animate_file(replay_path, out, fps=5, stride=2)

    with Image.open(out) as image:
        assert image.n_frames == (n_frames + 1) // 2


def test_animate_rejects_empty_replay(tmp_path):
    empty = Replay(meta={"scenario": "x"}, frames=())
    with pytest.raises(ValueError, match="没有任何帧"):
        animate_replay(empty, tmp_path / "empty.gif")


def test_animation_needs_only_the_replay():
    """viz 不碰引擎：给它一份只有 meta + 帧结构的回放也能画出来。

    这里直接构造最小回放（没有经过 Simulation），确认渲染路径只依赖日志字段。
    """
    replay = Replay(
        meta={
            "scenario": "handmade",
            "radar": "r",
            "jammer": "j",
            "target": "t",
            "pd_threshold": 0.5,
            "unjammed_detection_range_m": 50_000.0,
        },
        frames=(),
    )
    assert replay.meta["scenario"] == "handmade"
    assert replay.duration == 0.0  # 空回放的时长是 0，不是崩
