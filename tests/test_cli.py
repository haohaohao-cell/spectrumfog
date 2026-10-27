"""CLI 端到端：一条命令跑完想定 → 指标 → 三张图。

对应 T0 验收："``spectrumfog run scenarios/01_duel.yaml`` 一条命令出三张图"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrumfog.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
DUEL_YAML = REPO_ROOT / "scenarios" / "01_duel.yaml"

FIGURE_NAMES = {
    "01_duel_burnthrough.png",
    "01_duel_pd_vs_range.png",
    "01_duel_js_vs_range.png",
}


def test_run_prints_report(capsys):
    assert main(["run", str(DUEL_YAML), "--no-figures"]) == 0

    out = capsys.readouterr().out
    assert "01_duel" in out
    assert "R_bt" in out
    assert "J/S" in out
    assert "Pd" in out
    assert "虚构" in out  # 合规红线：输出里必须带"参数为虚构值"


def test_run_writes_three_figures(tmp_path, capsys):
    assert main(["run", str(DUEL_YAML), "--out", str(tmp_path)]) == 0

    produced = {path.name for path in tmp_path.glob("*.png")}
    assert produced == FIGURE_NAMES
    for path in tmp_path.glob("*.png"):
        # 空白 / 只有坐标轴的图会明显小于 10 KB
        assert path.stat().st_size > 10_000
    assert "图" in capsys.readouterr().out


def test_run_missing_scenario_returns_1(tmp_path):
    assert main(["run", str(tmp_path / "nope.yaml")]) == 1


def test_run_wrong_jammer_id_returns_1():
    # radar_a 上没有 noise_jammer 辐射源
    assert main(["run", str(DUEL_YAML), "--jammer", "radar_a", "--no-figures"]) == 1


def test_run_unknown_radar_id_returns_1():
    assert main(["run", str(DUEL_YAML), "--radar", "ghost", "--no-figures"]) == 1


def test_burn_through_curve_is_single_scenario_marker(tmp_path):
    """烧穿曲线本身：干扰越强 R_bt 越小（单调性，不需要看图，看数据）。"""
    from spectrumfog.model.scenario import load_scenario
    from spectrumfog.physics import radar

    scenario = load_scenario(DUEL_YAML)
    params = scenario.link_params()
    rbt_kwargs = {k: v for k, v in params.items() if k != "lam"}

    weak = float(radar.burn_through_range(**{**rbt_kwargs, "pj_w": params["pj_w"] / 10}))
    strong = float(
        radar.burn_through_range(**{**rbt_kwargs, "pj_w": params["pj_w"] * 10})
    )
    assert strong < weak


def test_parser_requires_subcommand():
    parser_args = ["run"]
    with pytest.raises(SystemExit):
        main(parser_args)  # 缺 scenario 参数 → argparse 退出码 2


# --- sim / animate -----------------------------------------------------------

SHORT_YAML = """
meta: {name: short_standoff, seed: 3, dt: 2.0, duration: 10.0}
environment: {noise_figure_db: 3.0, temperature_k: 290}
platforms:
  - id: radar_a
    side: blue
    pos: [0, 0]
    emitters:
      - {kind: radar, freq_hz: 3.0e9, bw_hz: 2.0e6, pt_w: 1.5e5, gain_db: 35, sigma_m2: 1.0, loss_db: 3}
  - id: jammer_b
    side: red
    pos: [30000, 0]
    waypoints:
      - {t: 0, pos: [30000, 0]}
      - {t: 10, pos: [10000, 0]}
    emitters:
      - {kind: noise_jammer, freq_hz: 3.0e9, bw_hz: 1.0e8, pj_w: 2.0, gain_db: 0}
  - {id: target_c, side: red, pos: [12000, 4000], emitters: []}
objectives:
  - {type: detect, target: target_c}
"""


@pytest.fixture()
def short_scenario(tmp_path) -> Path:
    path = tmp_path / "short.yaml"
    path.write_text(SHORT_YAML, encoding="utf-8")
    return path


def test_sim_writes_replay_and_reports(short_scenario, tmp_path, capsys):
    assert main(["sim", str(short_scenario), "--out", str(tmp_path)]) == 0

    replay_path = tmp_path / "short_standoff.jsonl"
    assert replay_path.exists()
    lines = replay_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 6 * (3 + 1 + 1)  # meta + 6 帧 × (3 平台 + link + detection)

    out = capsys.readouterr().out
    assert "short_standoff" in out
    assert "replay" in out
    assert "虚构" in out


def test_sim_replay_is_reproducible(short_scenario, tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert main(["sim", str(short_scenario), "--replay", str(first)]) == 0
    assert main(["sim", str(short_scenario), "--replay", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_sim_with_gif(short_scenario, tmp_path):
    assert main(["sim", str(short_scenario), "--out", str(tmp_path), "--gif", "--fps", "6"]) == 0
    gif = tmp_path / "short_standoff.gif"
    assert gif.exists()
    assert gif.stat().st_size > 5_000


def test_animate_from_replay(short_scenario, tmp_path):
    assert main(["sim", str(short_scenario), "--out", str(tmp_path)]) == 0
    replay_path = tmp_path / "short_standoff.jsonl"
    out = tmp_path / "anim.gif"

    assert main(["animate", str(replay_path), "-o", str(out), "--fps", "6"]) == 0
    assert out.exists()


def test_animate_missing_replay_returns_1(tmp_path):
    assert main(["animate", str(tmp_path / "nope.jsonl")]) == 1
