"""诱饵（转发式假目标）：等效 RCS、回波功率、事件与真实目标的关系。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario, parse_scenario
from spectrumfog.physics import radar

REPO_ROOT = Path(__file__).resolve().parents[1]
DECOY_YAML = REPO_ROOT / "scenarios" / "06_decoy.yaml"


def _scenario(**overrides) -> dict:
    base = {
        "meta": {"name": "decoy_case", "seed": 3, "dt": 2.0, "duration": 10.0},
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
                        "gain_db": 35,
                        "sigma_m2": 2.0,
                        "loss_db": 3,
                    }
                ],
            },
            {"id": "target_r", "side": "red", "pos": [70000, 10000], "emitters": []},
            {
                "id": "decoy_r",
                "side": "red",
                "pos": [22000, -3000],
                "emitters": [
                    {
                        "kind": "decoy",
                        "freq_hz": 3.0e9,
                        "bw_hz": 2.0e6,
                        "gain_db": 3,
                        "amplification_db": 30,
                    }
                ],
            },
            {
                "id": "jammer_w",
                "side": "red",
                "pos": [90000, 20000],
                "emitters": [
                    {
                        "kind": "noise_jammer",
                        "freq_hz": 3.0e9,
                        "bw_hz": 1.0e8,
                        "pj_w": 0.5,
                        "gain_db": 0,
                    }
                ],
            },
        ],
        "sensors": [
            {"id": "esm_1", "kind": "esm", "side": "blue", "pos": [0, -25000],
             "noise_figure_db": 6.0, "bw_hz": 1.0e5},
            {"id": "esm_2", "kind": "esm", "side": "blue", "pos": [0, 25000],
             "noise_figure_db": 6.0, "bw_hz": 1.0e5},
        ],
        "objectives": [
            {"type": "detect", "target": "target_r"},
            {"type": "localize", "target": "decoy_r", "cep_m": 500},
        ],
    }
    base.update(overrides)
    return base


def test_equivalent_rcs_matches_the_echo_power():
    """σ_eq 的定义就是"同距离上给出同样回波功率的目标 RCS"。"""
    lam, gain, amp, loss = 0.1, 2.0, 1000.0, 2.0
    sigma_eq = float(radar.decoy_equivalent_rcs(lam, gain, amp, loss))

    for range_m in (5e3, 2.2e4, 8e4):
        echo = float(
            radar.decoy_echo_power(1e5, 3162.0, 3162.0, gain, amp, lam, range_m)
        )
        equivalent = float(
            radar.received_power(
                pt_w=1e5,
                gt=3162.0,
                gr=3162.0,
                lam=lam,
                sigma_m2=sigma_eq,
                range_m=range_m,
                loss=loss,
            )
        )
        assert echo == pytest.approx(equivalent, rel=1e-12)


def test_decoy_rcs_scales_with_amplification():
    base = float(radar.decoy_equivalent_rcs(0.1, 2.0, 100.0, 2.0))
    assert float(radar.decoy_equivalent_rcs(0.1, 2.0, 1000.0, 2.0)) == pytest.approx(
        10.0 * base, rel=1e-12
    )


def test_decoy_transmit_power_is_illumination_times_amplification():
    lam, gain, amp, range_m = 0.1, 2.0, 100.0, 15000.0
    tx = float(radar.decoy_transmit_power(1e5, 3162.0, gain, amp, lam, range_m))
    illumination = 1e5 * 3162.0 * gain * lam**2 / ((4 * np.pi) ** 2 * range_m**2)
    assert tx == pytest.approx(illumination * amp, rel=1e-12)


def test_sim_emits_decoy_events_with_a_brightness_ratio():
    result = Simulation(parse_scenario(_scenario())).run()
    decoys = [event for event in result.events if event["kind"] == "decoy"]

    assert decoys, "应该产生 decoy 事件"
    first = decoys[0]
    assert first["decoy"] == "decoy_r"
    assert first["radar"] == "radar_a"
    assert first["sigma_eq_m2"] > 0.0
    # 诱饵在 22 km、等效 6 m²；真目标在 70 km、2 m² —— 诱饵应该亮得多
    assert first["echo_ratio"] > 10.0
    assert 0.0 <= first["pd"] <= 1.0
    assert result.summary["decoy_detected_fraction"] is not None


def test_decoy_is_invisible_to_a_different_band():
    data = _scenario()
    data["platforms"][1]["emitters"] = []
    data["platforms"][2]["emitters"][0]["freq_hz"] = 3.0e8  # 挪到通信频段
    result = Simulation(parse_scenario(data)).run()

    assert [e for e in result.events if e["kind"] == "decoy"] == []
    assert result.summary["decoy_detected_fraction"] is None


def test_decoy_scenario_story():
    """06_decoy：真目标离得远、诱饵更亮，雷达主要看到的是诱饵。"""
    result = Simulation(load_scenario(DECOY_YAML)).run()
    decoys = [e for e in result.events if e["kind"] == "decoy"]
    links = [e for e in result.events if e["kind"] == "link"]

    assert decoys
    assert decoys[0]["echo_ratio"] > 50.0
    assert result.summary["decoy_detected_fraction"] > 0.5
    # 真目标 70 km 外、又被弱干扰机压着 —— Pd 远低于诱饵
    assert links[0]["pd"] < decoys[0]["pd"]
