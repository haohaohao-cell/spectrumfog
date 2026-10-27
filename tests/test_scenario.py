"""想定加载：单位换算、校验、以及从想定到物理参数的那一段胶水。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectrumfog.model.scenario import (
    load_scenario,
    normalize_units,
    parse_scenario,
)
from spectrumfog.physics import radar

REPO_ROOT = Path(__file__).resolve().parents[1]
DUEL_YAML = REPO_ROOT / "scenarios" / "01_duel.yaml"


def _duel_dict() -> dict:
    return {
        "meta": {"name": "tiny", "seed": 7, "dt": 1.0, "duration": 10.0},
        "platforms": [
            {
                "id": "r1",
                "side": "blue",
                "pos": [0.0, 0.0],
                "emitters": [
                    {
                        "kind": "radar",
                        "freq_hz": 3.0e9,
                        "bw_hz": 2.0e6,
                        "pt_w": 1.0e5,
                        "gain_db": 30,
                        "sigma_m2": 1.0,
                        "loss_db": 0,
                    }
                ],
            },
            {
                "id": "j1",
                "side": "red",
                "pos": [1.0e4, 0.0],
                "emitters": [
                    {
                        "kind": "noise_jammer",
                        "freq_hz": 3.0e9,
                        "bw_hz": 5.0e7,
                        "pj_w": 100.0,
                        "gain_db": 10,
                    }
                ],
            },
        ],
    }


# --- 01_duel -----------------------------------------------------------------


def test_load_duel_scenario_converts_to_si():
    """01_duel.yaml 加载后：dB → 线性、Hz/W/m 就是 SI 原值。"""
    scenario = load_scenario(DUEL_YAML)

    assert scenario.meta.name == "01_duel"
    assert scenario.meta.seed == 42
    assert scenario.meta.dt == pytest.approx(1.0)
    assert scenario.meta.duration == pytest.approx(600.0)

    # 环境：3 dB 噪声系数 → 1.995 线性
    assert scenario.environment.noise_figure == pytest.approx(10 ** 0.3, rel=1e-12)
    assert scenario.environment.temperature_k == pytest.approx(290.0)

    radar_platform, radar_emitter = scenario.primary_radar()
    jammer_platform, jammer_emitter = scenario.primary_jammer()

    assert radar_platform.id == "radar_a"
    assert radar_emitter.freq_hz == pytest.approx(3.0e9)
    assert radar_emitter.gain == pytest.approx(10**3.5, rel=1e-12)  # 35 dBi
    assert radar_emitter.loss == pytest.approx(10**0.3, rel=1e-12)  # 3 dB
    assert jammer_platform.id == "jammer_b"
    assert jammer_emitter.pj_w == pytest.approx(90.0)
    assert jammer_emitter.gain == pytest.approx(1.0)  # 0 dBi
    assert scenario.standoff_m() == pytest.approx(30_000.0)


def test_duel_link_params_drive_physics_end_to_end():
    """从 YAML 一路到物理量：烧穿距离落在 km 量级、站距处被压制。"""
    scenario = load_scenario(DUEL_YAML)
    params = scenario.link_params()

    r_bt = float(
        radar.burn_through_range(**{k: v for k, v in params.items() if k != "lam"})
    )
    js = float(radar.js_ratio_self_screen(range_m=scenario.standoff_m(), **params))

    assert 1.0e3 < r_bt < 1.0e4  # 1–10 km
    assert js > 1.0  # 30 km 处干扰占优
    # 站距 > R_bt ⇒ 被压制；这是 01_duel 想定要讲的唯一一件事
    assert scenario.standoff_m() > r_bt


# --- 单位归一化 --------------------------------------------------------------


def test_unit_suffixes_are_converted():
    """freq_mhz / pt_dbm / pj_kw / pos_km / dt_ms 都要在加载时换成 SI。"""
    data = _duel_dict()
    data["meta"] = {"name": "units", "dt_ms": 500, "duration_ms": 2000}
    data["platforms"][0].pop("pos")
    data["platforms"][0]["pos_km"] = [0, 0]
    data["platforms"][1]["pos"] = [10_000, 0]
    data["platforms"][0]["emitters"][0] = {
        "kind": "radar",
        "freq_mhz": 3000,
        "bw_mhz": 2,
        "pt_dbm": 80,
        "gain_db": 30,
        "sigma_m2": 1.0,
        "loss_db": 0,
    }
    data["platforms"][1]["emitters"][0] = {
        "kind": "noise_jammer",
        "freq_ghz": 3.0,
        "bw_mhz": 50,
        "pj_kw": 0.1,
        "gain_db": 10,
    }

    scenario = parse_scenario(data)
    _, radar_emitter = scenario.primary_radar()
    _, jammer_emitter = scenario.primary_jammer()

    assert scenario.meta.dt == pytest.approx(0.5)
    assert scenario.meta.duration == pytest.approx(2.0)
    assert radar_emitter.freq_hz == pytest.approx(3.0e9)  # 3000 MHz
    assert radar_emitter.pt_w == pytest.approx(1.0e5)  # 80 dBm
    assert jammer_emitter.pj_w == pytest.approx(100.0)  # 0.1 kW
    assert jammer_emitter.bw_hz == pytest.approx(5.0e7)  # 50 MHz
    assert scenario.platform("r1").pos == pytest.approx((0.0, 0.0))


def test_normalize_units_rejects_two_units_for_one_quantity():
    with pytest.raises(ValueError, match="同一个量"):
        normalize_units({"freq_hz": 3.0e9, "freq_mhz": 3000.0})


def test_normalize_units_walks_nested_lists():
    data = {"platforms": [{"pos_km": [30, 0]}, {"pos_nmi": [1, 0]}]}
    normalized = normalize_units(data)
    assert normalized["platforms"][0]["pos"] == pytest.approx([30_000.0, 0.0])
    assert normalized["platforms"][1]["pos"] == pytest.approx([1852.0, 0.0])


# --- 校验 --------------------------------------------------------------------


def test_duplicate_platform_ids_rejected():
    data = _duel_dict()
    data["platforms"][1]["id"] = "r1"
    with pytest.raises(ValueError, match="重复"):
        parse_scenario(data)


def test_objective_must_point_to_existing_platform():
    data = _duel_dict()
    data["objectives"] = [{"type": "localize", "target": "ghost", "cep_m": 500}]
    with pytest.raises(ValueError, match="不存在的平台"):
        parse_scenario(data)


def test_unknown_field_rejected():
    data = _duel_dict()
    data["platforms"][0]["emitters"][0]["mystery"] = 1
    with pytest.raises(ValidationError):
        parse_scenario(data)


def test_scenario_is_frozen():
    scenario = parse_scenario(_duel_dict())
    with pytest.raises(ValidationError):
        scenario.meta.name = "changed"  # type: ignore[misc]


def test_missing_scenario_file_raises():
    with pytest.raises(FileNotFoundError):
        load_scenario(REPO_ROOT / "scenarios" / "does_not_exist.yaml")


def test_empty_yaml_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空"):
        load_scenario(path)


def test_missing_jammer_raises():
    data = _duel_dict()
    data["platforms"] = data["platforms"][:1]  # 只剩雷达
    scenario = parse_scenario(data)
    with pytest.raises(ValueError, match="noise_jammer"):
        scenario.primary_jammer()
