"""想定集合、提交物与排行榜。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spectrumfog.bench.runner import run_bench
from spectrumfog.bench.suite import (
    SUBMISSION_SCHEMA,
    build_submission,
    load_suite,
    render_leaderboard,
    verify_submission,
)
from spectrumfog.cli import main
from spectrumfog.model.scenario import parse_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_V1 = REPO_ROOT / "scenarios" / "suite_v1.yaml"


def _scenario_dict() -> dict:
    return {
        "meta": {"name": "leaderboard_case", "seed": 1, "dt": 5.0, "duration": 10.0},
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
            {"id": "target_c", "side": "red", "pos": [12000, 4000], "emitters": []},
        ],
        "sensors": [
            {"id": "esm_1", "kind": "esm", "side": "blue", "pos": [0, -25000],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
            {"id": "esm_2", "kind": "esm", "side": "blue", "pos": [0, 25000],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
        ],
        "objectives": [
            {"type": "detect", "target": "target_c"},
            {"type": "localize", "target": "jammer_b", "cep_m": 500},
        ],
    }


def _write_suite(tmp_path: Path, scenario_path: Path, **overrides) -> Path:
    payload = {
        "version": 1,
        "name": "test-suite",
        "created": "2026-10-27",
        "seeds": [0, 1],
        "policies": ["random"],
        "scenarios": [str(scenario_path)],
    }
    payload.update(overrides)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def scenario_path(tmp_path: Path) -> Path:
    path = tmp_path / "case.yaml"
    path.write_text(
        yaml.safe_dump(_scenario_dict(), allow_unicode=True), encoding="utf-8"
    )
    return path


def test_repository_suite_is_v1(scenario_path):
    suite = load_suite(SUITE_V1)

    assert suite.version == 1
    assert suite.name == "spectrumfog-v1"
    assert len(suite.scenario_paths) >= 6
    assert len(suite.seeds) >= 3
    assert set(suite.policies) == {"random", "always-jam", "stealth"}
    assert suite.episodes == len(suite.seeds)


def test_suite_hashes_follow_the_content_not_the_bytes(tmp_path, scenario_path):
    suite = load_suite(_write_suite(tmp_path, scenario_path))
    before = suite.scenario_hashes()

    # 只改注释 / 空行：内容不变 → 指纹不变
    scenario_path.write_text(
        "# a comment\n\n" + scenario_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert suite.scenario_hashes() == before

    # 改一个物理参数：指纹必须变
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    payload["platforms"][0]["emitters"][0]["pt_w"] = 2.0e5
    scenario_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    assert suite.scenario_hashes() != before


def test_suite_rejects_missing_fields(tmp_path, scenario_path):
    path = _write_suite(tmp_path, scenario_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["seeds"]
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="seeds"):
        load_suite(path)


def test_submission_roundtrip_and_verify(tmp_path, scenario_path):
    suite = load_suite(_write_suite(tmp_path, scenario_path))
    scenario = parse_scenario(_scenario_dict())
    rows = run_bench([scenario], ["random"], seeds=suite.seeds)

    payload = build_submission(suite, rows, label="tester")
    assert payload["schema"] == SUBMISSION_SCHEMA
    assert payload["suite"]["version"] == 1
    assert len(payload["rows"]) == 1
    assert payload["score"] == pytest.approx(rows[0].reward_mean)
    assert verify_submission(payload, suite) == []

    # 现场改一个参数 → 内容指纹对不上
    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    data["platforms"][0]["emitters"][0]["pt_w"] = 2.0e5
    scenario_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    problems = verify_submission(payload, suite)
    assert any("内容已改动" in problem for problem in problems)


def test_verify_detects_suite_version_drift(tmp_path, scenario_path):
    suite = load_suite(_write_suite(tmp_path, scenario_path))
    scenario = parse_scenario(_scenario_dict())
    rows = run_bench([scenario], ["random"], seeds=suite.seeds)
    payload = build_submission(suite, rows, label="tester")

    newer = load_suite(_write_suite(tmp_path, scenario_path, version=2))
    problems = verify_submission(payload, newer)
    assert any("版本不符" in problem for problem in problems)


def test_verify_detects_missing_rows(tmp_path, scenario_path):
    suite = load_suite(_write_suite(tmp_path, scenario_path))
    scenario = parse_scenario(_scenario_dict())
    rows = run_bench([scenario], ["random"], seeds=suite.seeds)
    payload = build_submission(suite, rows, label="tester")
    payload["rows"] = payload["rows"][:0]

    assert any("行数不符" in problem for problem in verify_submission(payload, suite))


def test_leaderboard_sorts_and_renders():
    entries = [
        {"schema": SUBMISSION_SCHEMA, "label": "slow", "score": 1.0,
         "rows": [{"scenario": "a"}], "suite": {"version": 1},
         "environment": {"python": "3.12", "gymnasium": "1.4"}},
        {"schema": SUBMISSION_SCHEMA, "label": "fast", "score": 9.0,
         "rows": [{"scenario": "a"}], "suite": {"version": 1},
         "environment": {"python": "3.12", "gymnasium": "1.4"}},
    ]
    markdown = render_leaderboard(entries)

    assert markdown.startswith("# 排行榜")
    assert markdown.index("`fast`") < markdown.index("`slow`")


def test_leaderboard_handles_no_submissions():
    assert "还没有提交" in render_leaderboard([])


def test_cli_submit_and_verify(tmp_path, scenario_path):
    suite_path = _write_suite(tmp_path, scenario_path)
    submissions = tmp_path / "submissions"

    code = main(
        [
            "bench",
            "--suite",
            str(suite_path),
            "--submit",
            "--label",
            "cli-test",
            "--submissions",
            str(submissions),
        ]
    )
    assert code == 0
    files = list(submissions.glob("*.json"))
    assert len(files) == 1

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["label"] == "cli-test"

    assert main(["bench", "--verify", str(files[0]), "--suite", str(suite_path)]) == 0

    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    data["platforms"][1]["emitters"][0]["pj_w"] = 900.0
    scenario_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["bench", "--verify", str(files[0]), "--suite", str(suite_path)]) == 1


def test_cli_leaderboard_prints(tmp_path, capsys):
    assert main(["bench", "--leaderboard", str(tmp_path)]) == 0
    assert "还没有提交" in capsys.readouterr().out
