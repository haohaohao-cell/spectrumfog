"""通信链路模型与频段判据（T3b）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario, parse_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAD_YAML = REPO_ROOT / "scenarios" / "03_triad.yaml"


def _base() -> dict:
    return {
        "meta": {"name": "comms_case", "seed": 5, "dt": 2.0, "duration": 10.0},
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
                        "pt_w": 1.0e5,
                        "gain_db": 33,
                        "sigma_m2": 1.0,
                        "loss_db": 3,
                    }
                ],
            },
            {
                "id": "comms_a",
                "side": "blue",
                "pos": [0, 0],
                "emitters": [
                    {
                        "kind": "comms",
                        "freq_hz": 3.0e8,
                        "bw_hz": 2.5e4,
                        "ptx_w": 20.0,
                        "gain_db": 3,
                        "bit_rate_bps": 16000,
                        "to_platform": "comms_b",
                        "processing_gain_db": 6.0,
                    }
                ],
            },
            {"id": "comms_b", "side": "blue", "pos": [20000, 0], "emitters": []},
            {
                "id": "jammer_b",
                "side": "red",
                "pos": [30000, 0],
                "emitters": [
                    {
                        "kind": "noise_jammer",
                        "freq_hz": 3.0e8,
                        "bw_hz": 2.0e5,
                        "pj_w": 300.0,
                        "gain_db": 3,
                    }
                ],
            },
        ],
        "objectives": [{"type": "protect_link", "target": "comms_a"}],
    }


def test_comms_emitter_requires_a_real_receiver():
    data = _base()
    data["platforms"][1]["emitters"][0]["to_platform"] = "ghost"
    with pytest.raises(ValueError, match="不存在的平台"):
        parse_scenario(data)


def test_comms_emitter_rejects_self_link():
    data = _base()
    data["platforms"][1]["emitters"][0]["to_platform"] = "comms_a"
    with pytest.raises(ValueError, match="指向了自己"):
        parse_scenario(data)


def test_in_band_jammer_denies_the_link():
    result = Simulation(parse_scenario(_base())).run()
    comms = [event for event in result.events if event["kind"] == "comms"]

    assert comms, "应该产生 comms 事件"
    assert comms[0]["js"] > 1.0  # 300 W 打 25 kHz 链路，J/S 远大于 1
    assert comms[0]["aj_margin_db"] < 0.0  # 处理增益只有 6 dB，顶不住
    assert comms[0]["available"] is False
    assert result.summary["link_avail"] == pytest.approx(0.0)


def test_out_of_band_jammer_does_not_touch_the_link():
    data = _base()
    data["platforms"][3]["emitters"][0]["freq_hz"] = 3.0e9  # 干扰机换到雷达频段
    result = Simulation(parse_scenario(data)).run()
    comms = [event for event in result.events if event["kind"] == "comms"]

    assert comms[0]["js"] == pytest.approx(0.0)
    assert comms[0]["available"] is True
    assert result.summary["link_avail"] == pytest.approx(1.0)


def test_out_of_band_jammer_does_not_touch_the_radar():
    """同一个 90 W/300 W 干扰机，换到通信频段后对雷达完全无效。"""
    in_band = _base()
    in_band["platforms"][3]["emitters"][0].update(
        {"freq_hz": 3.0e9, "pj_w": 300.0}
    )
    out_band = _base()
    out_band["platforms"][3]["emitters"][0].update(
        {"freq_hz": 3.0e8, "pj_w": 300.0}
    )

    jamming = Simulation(parse_scenario(in_band)).run()
    clean = Simulation(parse_scenario(out_band)).run()
    jamming_links = [e for e in jamming.events if e["kind"] == "link"]
    clean_links = [e for e in clean.events if e["kind"] == "link"]

    assert all(event["js"] == pytest.approx(0.0) for event in clean_links)
    assert jamming_links[0]["js"] > 1.0
    # 同频时 Pd 被压下去，异频时恢复正常
    assert jamming_links[0]["pd"] < 0.01
    assert clean_links[0]["pd"] > 0.5


def test_link_availability_is_reported_in_the_summary():
    result = Simulation(load_scenario(TRIAD_YAML)).run()
    comms = [event for event in result.events if event["kind"] == "comms"]

    assert result.summary["link_avail"] is not None
    assert result.summary["n_comms_samples"] == len(comms)
    # 这一份想定里干扰机是边逼近边压制：链路会中途失效，不是全程失效
    assert 0.0 < result.summary["link_avail"] < 1.0
    assert comms[0]["available"] is True
    assert comms[-1]["available"] is False


def test_replay_carries_comms_frames():
    from spectrumfog.replay import Replay, build_frames

    result = Simulation(load_scenario(TRIAD_YAML)).run()
    frames = build_frames([e for e in result.events if e["kind"] != "meta"])

    assert frames[0].comms, "第一帧应该带 comms 事件"
    assert {"emitter", "receiver", "js", "aj_margin_db", "available"} <= set(
        frames[0].comms[0]
    )
