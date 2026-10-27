"""AAR：结论句、留一法归因、HTML 报告。"""

from __future__ import annotations

import re

import pytest

from spectrumfog import aar
from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import parse_scenario
from spectrumfog.replay import Replay, write_replay


def _scenario(**overrides) -> dict:
    base = {
        "meta": {"name": "aar_case", "seed": 3, "dt": 1.0, "duration": 20.0},
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
        "sensors": [
            {"id": "esm_1", "kind": "esm", "side": "blue", "pos": [0, -25000],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
            {"id": "esm_2", "kind": "esm", "side": "blue", "pos": [0, 25000],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
            {"id": "esm_3", "kind": "esm", "side": "blue", "pos": [-25000, 0],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
        ],
        "objectives": [{"type": "localize", "target": "jammer_b", "cep_m": 500}],
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def replay(tmp_path_factory) -> Replay:
    scenario = parse_scenario(_scenario())
    result = Simulation(scenario).run()
    path = tmp_path_factory.mktemp("aar") / "case.jsonl"
    write_replay(path, result.events)
    return Replay.load(path)


def test_report_answers_when_and_why(replay):
    report = aar.analyse(replay)

    assert report.emitter == "jammer_b"
    assert report.n_fixes > 0
    assert report.first_effective_s is not None
    # 验收要求的形状：时间 + 原因
    assert re.search(r"t = \d+ s 被有效定位", report.sentence)
    assert "esm_" in report.sentence
    assert "门槛" in report.sentence


def test_attribution_is_leave_one_out(replay):
    report = aar.analyse(replay)
    sensors = {row["sensor"] for row in report.attribution}

    assert "esm_1" in sensors and "esm_3" in sensors
    for row in report.attribution:
        assert "verdict" in row
        # 去掉一台要么误差变大，要么根本解不出来
        assert row["error_m"] is None or row["error_m"] > 0


def test_threshold_changes_the_first_effective_moment(replay):
    loose = aar.analyse(replay, threshold_m=10_000.0)
    strict = aar.analyse(replay, threshold_m=1.0)

    assert loose.first_effective_s is not None
    assert loose.first_effective_s <= (strict.first_effective_s or 10**9)
    if strict.first_effective_s is None:
        assert "全程没有被有效定位" in strict.sentence


def test_single_sensor_cannot_localise(tmp_path):
    scenario = parse_scenario(
        _scenario(
            sensors=[
                {"id": "esm_1", "kind": "esm", "side": "blue", "pos": [0, -25000],
                 "noise_figure_db": 12.0, "bw_hz": 1.0e6}
            ]
        )
    )
    result = Simulation(scenario).run()
    path = tmp_path / "single.jsonl"
    write_replay(path, result.events)
    report = aar.analyse(Replay.load(path))

    assert report.n_fixes == 0
    assert "全程没有被有效定位" in report.sentence
    assert report.attribution == ()


def test_html_contains_the_verdict_and_the_numbers(replay):
    report = aar.analyse(replay)
    document = aar.render_html(report, replay)

    assert document.startswith("<!doctype html>")
    assert "AAR" in document
    assert report.emitter in document
    assert "留一法" in document
    assert "<svg" in document
    # 自包含：不引用外部资源
    assert "http://" not in document and "https://" not in document


def test_report_is_reproducible(replay):
    first = aar.analyse(replay)
    second = aar.analyse(replay)
    assert first.sentence == second.sentence
    assert first.cep_series == second.cep_series
