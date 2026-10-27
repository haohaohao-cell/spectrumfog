"""Web 数据包：二维表编码、与物理层一致、1D 曲线自洽。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectrumfog.cli import main
from spectrumfog.model.scenario import load_scenario
from spectrumfog.physics import radar
from spectrumfog.webexport import (
    SCHEMA,
    build_web_bundle,
    pack_array,
    unpack_array,
    write_web_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDOFF_YAML = REPO_ROOT / "scenarios" / "02_standoff.yaml"


@pytest.fixture(scope="module")
def bundle() -> dict:
    return build_web_bundle(load_scenario(STANDOFF_YAML), samples=33, ring_samples=61)


def test_pack_unpack_roundtrip_is_lossless_for_float32():
    values = np.array([[0.0, 0.5], [1.0, 1.0e6]], dtype=np.float64)
    packed = pack_array(values)
    restored = unpack_array(packed)

    assert packed["dtype"] == "float32"
    assert packed["shape"] == [2, 2]
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, values, rtol=1e-7)


def test_bundle_has_documented_schema(bundle):
    assert bundle["schema"] == SCHEMA
    assert bundle["units"].startswith("SI")
    for key in ("radar", "jammer", "target", "grid", "pd", "js", "rings", "target_curves"):
        assert key in bundle

    grid = bundle["grid"]
    assert grid["spacing"] == "log10"
    assert grid["rt_m"]["n"] == 33
    assert grid["rj_m"]["n"] == 33
    assert bundle["pd"]["shape"] == [33, 33]
    assert bundle["js"]["shape"] == [33, 33]


def test_table_matches_physics_at_grid_points(bundle):
    """表里任意一格都必须等于物理层的直接计算（float32 精度内）。

    这是"前端不重写物理"的底线：表不是另算的一套，就是同一函数的采样。
    """
    scenario = load_scenario(STANDOFF_YAML)
    params = scenario.link_params()
    environment = scenario.environment
    pn_w = float(
        radar.noise_power(
            bw_hz=params["bw_r_hz"],
            noise_figure=environment.noise_figure,
            temperature_k=environment.temperature_k,
        )
    )

    grid = bundle["grid"]
    rt = np.geomspace(grid["rt_m"]["min"], grid["rt_m"]["max"], grid["rt_m"]["n"])
    rj = np.geomspace(grid["rj_m"]["min"], grid["rj_m"]["max"], grid["rj_m"]["n"])
    pd = unpack_array(bundle["pd"])
    js = unpack_array(bundle["js"])

    for i, j in ((0, 0), (10, 20), (16, 16), (32, 32)):
        pr = radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=rt[j],
            loss=params["loss"],
        )
        pj_rx = radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=rj[i],
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
        expected_pd = float(radar.pd_from_powers(bundle["pfa"], pr, pn_w, pj_rx))
        expected_js = float(
            radar.js_ratio_standoff(
                range_m=rt[j],
                jammer_range_m=rj[i],
                **{key: params[key] for key in ("pt_w", "gain", "sigma_m2", "lam",
                                                "pj_w", "gj", "bw_r_hz", "bw_j_hz", "loss")},
            )
        )
        assert pd[i, j] == pytest.approx(expected_pd, rel=1e-5, abs=1e-7)
        assert js[i, j] == pytest.approx(expected_js, rel=1e-5)


def test_detection_ring_grows_with_jammer_range(bundle):
    """干扰机越远，检测边界越大（T2 拖动时环跟着放大）。"""
    rj = np.asarray(bundle["rings"]["jammer_range_m"])
    detection = np.asarray(bundle["rings"]["detection_m"])
    assert np.all(np.diff(detection) > 0)

    # 环半径始终小于 JS=1 的烧穿半径：检测比"烧穿"更严格（docs/physics.md §7.2）
    burn = np.asarray(bundle["rings"]["burn_through_m"])
    assert np.all(detection < burn)


def test_burn_through_ring_follows_sqrt_law(bundle):
    rj = np.asarray(bundle["rings"]["jammer_range_m"])
    burn = np.asarray(bundle["rings"]["burn_through_m"])
    params = load_scenario(STANDOFF_YAML).link_params()
    r_bt_self = float(
        radar.burn_through_range(
            **{key: params[key] for key in ("pt_w", "gain", "sigma_m2", "pj_w",
                                            "gj", "bw_r_hz", "bw_j_hz", "loss")}
        )
    )
    np.testing.assert_allclose(burn, np.sqrt(r_bt_self * rj), rtol=1e-5)


def test_target_curves_match_physics(bundle):
    scenario = load_scenario(STANDOFF_YAML)
    params = scenario.link_params()
    curves = bundle["target_curves"]
    rj = np.asarray(curves["jammer_range_m"])
    target_range = float(bundle["target"]["range_m"])

    for index in (0, len(rj) // 2, len(rj) - 1):
        expected = float(
            radar.js_ratio_standoff(
                range_m=target_range,
                jammer_range_m=rj[index],
                **{key: params[key] for key in ("pt_w", "gain", "sigma_m2", "lam",
                                                "pj_w", "gj", "bw_r_hz", "bw_j_hz", "loss")},
            )
        )
        assert curves["js"][index] == pytest.approx(expected, rel=1e-9)


def test_bundle_is_json_serialisable(bundle, tmp_path):
    path = tmp_path / "data.json"
    write_web_bundle(path, bundle)

    raw = path.read_text(encoding="utf-8")
    assert "\n" in raw  # 尾部换行
    restored = json.loads(raw)
    assert restored["schema"] == SCHEMA
    np.testing.assert_allclose(
        unpack_array(restored["pd"]), unpack_array(bundle["pd"]), rtol=1e-7
    )


def test_bundle_rejects_bad_grid_samples():
    scenario = load_scenario(STANDOFF_YAML)
    with pytest.raises(ValueError, match="samples"):
        build_web_bundle(scenario, samples=2)


def test_cli_export_web_writes_file(tmp_path, capsys):
    out = tmp_path / "bundle.json"
    assert main(["export-web", str(STANDOFF_YAML), "--out", str(out)]) == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario"] == "02_standoff"
    assert "二维表" in capsys.readouterr().out
