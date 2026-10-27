"""Gymnasium 环境、baseline 策略与评测器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from spectrumfog.bench.env import POWER_LEVELS, SpectrumFogEnv, decode_action
from spectrumfog.bench.policies import (
    POLICIES,
    AlwaysJamPolicy,
    RandomPolicy,
    StealthPolicy,
    make_policy,
)
from spectrumfog.bench.runner import format_table, run_bench, run_episode, write_json
from spectrumfog.cli import main
from spectrumfog.model.scenario import parse_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scenario(**overrides) -> dict:
    base = {
        "meta": {"name": "bench_case", "seed": 1, "dt": 2.0, "duration": 20.0},
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
                "pos": [40000, 0],
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
            {"id": "esm_3", "kind": "esm", "side": "blue", "pos": [-25000, 0],
             "noise_figure_db": 12.0, "bw_hz": 1.0e6},
        ],
        "objectives": [
            {"type": "detect", "target": "target_c"},
            {"type": "localize", "target": "jammer_b", "cep_m": 500},
        ],
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def scenario():
    return parse_scenario(_scenario())


def test_env_follows_the_gym_api(scenario):
    env = SpectrumFogEnv(scenario)
    observation, info = env.reset(seed=3)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert observation.shape == (6,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    next_observation, reward, terminated, truncated, info = env.step(2)
    assert next_observation.shape == (6,)
    assert isinstance(float(reward), float)
    assert terminated is False and truncated is False
    assert {"pd", "cep", "localized", "js"} <= set(info)


def test_episode_ends_after_the_scenario_duration(scenario):
    env = SpectrumFogEnv(scenario)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        if terminated or truncated:
            break
    assert steps == int(round(scenario.meta.duration / scenario.meta.dt))  # 10


def test_power_off_means_no_emission_and_no_localisation(scenario):
    env = SpectrumFogEnv(scenario)
    env.reset(seed=0)
    # 动作 = 停 + 关机（方向 0，功率档 0）
    _, reward, _, _, info = env.step(0)

    assert info["power"] == 0.0
    assert info["cep"] != info["cep"]  # nan：没有辐射就没有量测
    assert info["localized"] is False
    assert reward < 1.0 + 1e-9


def test_same_seed_gives_the_same_trajectory(scenario):
    actions = [2, 2, 5, 8, 2, 14, 0, 11, 2, 3]

    def rollout() -> list[float]:
        env = SpectrumFogEnv(scenario)
        env.reset(seed=7)
        rewards = []
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        return rewards

    assert rollout() == rollout()


def test_action_decoding_covers_the_space():
    seen = {decode_action(action) for action in range(15)}
    assert len(seen) == 15
    assert decode_action(2) == (0, 2)
    assert decode_action(14) == (4, 2)
    with pytest.raises(ValueError):
        decode_action(15)


def test_policies_only_use_the_observation(scenario):
    env = SpectrumFogEnv(scenario)
    observation, _ = env.reset(seed=0)

    for policy_cls in POLICIES:
        policy = policy_cls()
        policy.reset(seed=0)
        action = policy.act(observation)
        assert 0 <= action < 15


def test_make_policy_rejects_unknown_names():
    with pytest.raises(ValueError, match="未知策略"):
        make_policy("nonexistent")
    assert isinstance(make_policy("stealth"), StealthPolicy)
    assert isinstance(make_policy("random"), RandomPolicy)
    assert isinstance(make_policy("always-jam"), AlwaysJamPolicy)


def test_always_jam_never_turns_off(scenario):
    policy = AlwaysJamPolicy()
    policy.reset(seed=0)
    for _ in range(5):
        assert decode_action(policy.act(np.zeros(6, dtype=np.float32))) == (0, 2)


def test_run_episode_reports_consistent_metrics(scenario):
    result = run_episode(scenario, AlwaysJamPolicy(), seed=0)
    assert result.steps == 10
    assert 0.0 <= result.exposure_fraction <= 1.0
    assert 0.0 <= result.pd_below_threshold_fraction <= 1.0
    assert result.first_warning_step is None or result.first_warning_step >= 1


def test_run_bench_builds_one_row_per_combination(scenario):
    rows = run_bench([scenario], ["random", "always-jam"], episodes=2)
    assert len(rows) == 2
    assert {row.policy for row in rows} == {"random", "always-jam"}
    # 同一想定内按奖励从高到低
    assert rows[0].reward_mean >= rows[1].reward_mean
    table = format_table(rows)
    assert "scenario" in table and "policy" in table


def test_bench_rejects_zero_episodes(scenario):
    with pytest.raises(ValueError):
        run_bench([scenario], ["random"], episodes=0)


def test_write_json_roundtrip(scenario, tmp_path):
    import json

    rows = run_bench([scenario], ["random"], episodes=1)
    path = write_json(tmp_path / "bench.json", rows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "spectrumfog.bench/1"
    assert payload["rows"][0]["scenario"] == "bench_case"


def test_cli_bench_runs_and_writes_markdown(tmp_path, capsys):
    yaml_path = tmp_path / "bench.yaml"
    import yaml as pyyaml

    yaml_path.write_text(
        pyyaml.safe_dump(_scenario(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    markdown = tmp_path / "board.md"
    code = main(
        [
            "bench",
            "--scenario",
            str(yaml_path),
            "--episodes",
            "1",
            "--markdown",
            str(markdown),
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "baseline 评测" in output
    assert "always-jam" in output
    assert markdown.exists()
    assert "| 想定 |" in markdown.read_text(encoding="utf-8")


def test_cli_bench_without_target_returns_2(capsys):
    assert main(["bench"]) == 2
