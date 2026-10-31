"""双人环境：EMCON 博弈（红方干扰机 vs 蓝方雷达）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrumfog.bench.duel import (
    BLUE_BASELINES,
    RED_BASELINES,
    DuelEnv,
    play_duel,
)
from spectrumfog.model.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
DUEL_YAML = REPO_ROOT / "scenarios" / "08_duel_2p.yaml"


@pytest.fixture(scope="module")
def scenario():
    return load_scenario(DUEL_YAML)


def test_env_has_two_action_spaces_and_seven_dim_observations(scenario):
    env = DuelEnv(scenario)
    observations, info = env.reset(seed=0)

    assert env.red_action_space.n == 15
    assert env.blue_action_space.n == 3
    assert observations["red"].shape == (7,)
    assert observations["blue"].shape == (7,)
    assert isinstance(info, dict)


def test_step_returns_two_rewards(scenario):
    env = DuelEnv(scenario)
    env.reset(seed=0)

    observations, rewards, terminated, truncated, info = env.step(2, 2)

    assert set(rewards) == {"red", "blue"}
    assert set(observations) == {"red", "blue"}
    assert terminated is False and truncated is False
    assert "pd" in info and "radiating" in info


def test_blue_off_means_no_link_and_no_exposure(scenario):
    env = DuelEnv(scenario)
    env.reset(seed=0)

    _, _, _, _, info = env.step(0, 0)  # 红方停+关机，蓝方关机

    assert info["radiating"] is False
    assert info["pd"] == 0.0
    assert info["blue_exposed"] is False
    assert info["red_exposed"] is False


def test_blue_on_is_tracked_when_red_keeps_looking(scenario):
    """蓝方一直开机：红方 ESM 应该能确认跟踪。"""
    env = DuelEnv(scenario)
    env.reset(seed=0)
    exposed = 0
    for step in range(30):
        _, _, _, _, info = env.step(2, 2)  # 红方满功率、蓝方满功率
        exposed += int(info["blue_exposed"])
    assert exposed > 0


def test_same_seed_gives_the_same_duel(scenario):
    def run() -> list[float]:
        env = DuelEnv(scenario)
        env.reset(seed=7)
        rewards = []
        for _ in range(25):
            _, reward, _, _, _ = env.step(2, 2)
            rewards.append(reward["red"] + reward["blue"])
        return rewards

    assert run() == run()


def test_play_duel_reports_both_sides(scenario):
    result = play_duel(
        scenario, red_policy="red-stay-jam", blue_policy="blue-pulse", episodes=1
    )

    assert result.episodes == 1
    assert 0.0 <= result.blue_on_fraction <= 1.0
    assert 0.0 <= result.blue_exposed_fraction <= 1.0
    assert 0.0 <= result.red_exposed_fraction <= 1.0
    assert set(result.as_row()) == {
        "blue",
        "red",
        "red_reward",
        "blue_reward",
        "blue_exposed",
        "red_exposed",
        "blue_on",
        "pd_mean",
    }


def test_pulsing_beats_staying_on_against_a_silent_red(scenario):
    """红方关机时，蓝方脉冲式开机应该能正分——这是 EMCON 的甜点。"""
    result = play_duel(
        scenario, red_policy="red-silent", blue_policy="blue-pulse", episodes=1
    )

    assert result.blue_reward > 0.0
    assert result.blue_exposed_fraction == pytest.approx(0.0)


def test_always_on_blue_is_punished(scenario):
    """同样红方关机，一直开机的蓝方被跟踪，分数应该低于脉冲式。"""
    always = play_duel(
        scenario, red_policy="red-silent", blue_policy="blue-always-on", episodes=1
    )
    pulse = play_duel(
        scenario, red_policy="red-silent", blue_policy="blue-pulse", episodes=1
    )

    assert always.blue_reward < pulse.blue_reward
    assert always.blue_exposed_fraction > pulse.blue_exposed_fraction


def test_baselines_are_registered():
    assert {policy.name for policy in RED_BASELINES} == {
        "red-stay-jam",
        "red-silent",
        "red-raider",
    }
    assert {policy.name for policy in BLUE_BASELINES} == {
        "blue-always-on",
        "blue-pulse",
        "blue-reactive",
    }
