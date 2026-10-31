"""自训练：线性策略 + CEM 的基本行为（不检验"学得好"，只检验"机制对"）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrumfog.bench.selfplay import (
    LinearSoftmax,
    features,
    format_history,
    self_play,
)
from spectrumfog.model.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
DUEL_YAML = REPO_ROOT / "scenarios" / "08_duel_2p.yaml"


@pytest.fixture(scope="module")
def scenario():
    return load_scenario(DUEL_YAML)


def test_features_add_time_basis():
    observation = np.zeros(7)
    expanded = features(observation)

    assert expanded.size > 7
    assert expanded.size == 7 + 2 * 4  # 4 个谐波，每个 sin+cos
    assert np.isfinite(expanded).all()


def test_policy_is_deterministic_without_rng():
    rng = np.random.default_rng(0)
    policy = LinearSoftmax.random(3, 7, rng)
    observation = np.full(7, 0.3)

    assert policy.act(observation) == policy.act(observation)


def test_policy_samples_different_actions_with_rng():
    rng = np.random.default_rng(0)
    policy = LinearSoftmax.random(3, 7, rng, scale=3.0)
    observation = np.zeros(7)
    choices = {
        policy.act(observation, rng=np.random.default_rng(seed)) for seed in range(50)
    }

    assert len(choices) > 1  # 有 rng 时按分布采样，不是恒定动作


def test_self_play_records_each_generation(scenario):
    history, red, blue = self_play(
        scenario, generations=2, iterations=2, population=2, seeds=(0,)
    )

    assert len(history) == 3  # 第 0 代基线 + 2 代
    assert [row["generation"] for row in history.generations] == [0, 1, 2]
    assert all(isinstance(row["red"], float) for row in history.generations)
    assert red.shape == (15, 7)
    assert blue.shape == (3, 7)
    assert "代" in format_history(history)


def test_self_play_is_reproducible(scenario):
    first, _, _ = self_play(
        scenario, generations=1, iterations=2, population=2, seeds=(0,), seed=3
    )
    second, _, _ = self_play(
        scenario, generations=1, iterations=2, population=2, seeds=(0,), seed=3
    )

    assert [row["red"] for row in first.generations] == [
        row["red"] for row in second.generations
    ]


def test_features_change_the_policy_shape(scenario):
    _, _, blue = self_play(
        scenario,
        generations=1,
        iterations=1,
        population=2,
        seeds=(0,),
        use_features=True,
    )

    assert blue.shape == (3, 7 + 2 * 4)
