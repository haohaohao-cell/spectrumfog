"""处理增益（`processing_gain_db`）：默认关闭，开启后把干扰机压回去。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrumfog.bench.env import SpectrumFogEnv
from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario, parse_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
DUEL = REPO_ROOT / "scenarios" / "01_duel.yaml"
GAIN = REPO_ROOT / "scenarios" / "07_gain.yaml"


def _one_link(path: Path) -> dict:
    result = Simulation(load_scenario(path)).run()
    return next(event for event in result.events if event["kind"] == "link")


def _scenario(gain_db: float):
    return {
        "meta": {"name": f"gain_{gain_db:g}", "seed": 1, "dt": 1.0, "duration": 5.0},
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
                        "processing_gain_db": gain_db,
                    }
                ],
            },
            {
                "id": "jammer_b",
                "side": "red",
                "pos": [30000, 0],
                "emitters": [
                    {
                        "kind": "noise_jammer",
                        "freq_hz": 3.0e9,
                        "bw_hz": 1.0e8,
                        "pj_w": 90.0,
                        "gain_db": 0,
                    }
                ],
            },
        ],
        "sensors": [],
        "objectives": [{"type": "localize", "target": "jammer_b", "cep_m": 500}],
    }


def test_default_is_off_so_legacy_numbers_do_not_move():
    link = _one_link(DUEL)

    assert link["processing_gain_db"] == 0.0
    assert link["js"] == pytest.approx(85.6315, rel=1e-4)
    assert link["pd"] == pytest.approx(1.173e-6, rel=1e-2)


def test_gain_scenario_is_the_same_geometry_with_40_db():
    duel = _one_link(DUEL)
    gain = _one_link(GAIN)

    assert gain["processing_gain_db"] == 40.0
    assert gain["range_target_m"] == pytest.approx(duel["range_target_m"])
    assert gain["range_jammer_m"] == pytest.approx(duel["range_jammer_m"])


def test_40_db_gain_flips_the_outcome():
    duel = _one_link(DUEL)
    gain = _one_link(GAIN)

    # J/S 除以 10^4
    assert gain["js"] == pytest.approx(duel["js"] / 1e4, rel=1e-6)
    # Pd 从"压死"变成"看得见"
    assert duel["pd"] < 1e-5
    assert gain["pd"] > 0.5


def test_20_db_gain_is_not_enough():
    """20 dB 只把 J/S 压到 0.86，Pd 仍是千分之几——远没翻盘。"""
    result = Simulation(parse_scenario(_scenario(20.0))).run()
    link = next(event for event in result.events if event["kind"] == "link")

    assert link["js"] == pytest.approx(0.8563, rel=1e-3)
    assert 1e-4 < link["pd"] < 0.01


def test_gain_scales_burn_through_as_expected():
    """自屏蔽 R_bt ∝ sqrt(Gp)：40 dB 时 3.24 km → 324 km。"""
    duel = Simulation(load_scenario(DUEL)).run()
    gain = Simulation(load_scenario(GAIN)).run()

    assert gain.events[0]["burn_through_self_m"] == pytest.approx(
        duel.events[0]["burn_through_self_m"] * 100.0, rel=1e-6
    )


def test_gain_is_recorded_in_meta_and_link_events():
    result = Simulation(load_scenario(GAIN)).run()

    assert result.events[0]["processing_gain_db"] == 40.0
    assert all(
        event["processing_gain_db"] == 40.0
        for event in result.events
        if event["kind"] == "link"
    )


def test_bench_env_honours_processing_gain():
    scenario = load_scenario(GAIN)
    env = SpectrumFogEnv(scenario)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0 * 3 + 2)  # 停 + 满功率

    assert info["js"] < 0.05  # 满功率也只有 0.0086 的 J/S
    assert reward < 1.0  # 压不住 → 拿不到"干扰有效"的分


def test_gain_does_not_affect_the_unjammed_range():
    duel = Simulation(load_scenario(DUEL)).run()
    gain = Simulation(load_scenario(GAIN)).run()

    assert gain.events[0]["unjammed_detection_range_m"] == pytest.approx(
        duel.events[0]["unjammed_detection_range_m"]
    )
