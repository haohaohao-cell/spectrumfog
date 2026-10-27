"""评测器：策略 × 想定 × 种子 → 一张可比表格。

同一 seed 下，所有策略看到的是**同一串随机数**（同一个初始种子派生），
所以差异来自决策，不来自运气。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..model.scenario import Scenario
from .policies import Policy, make_policy

__all__ = ["EpisodeResult", "BenchRow", "run_episode", "run_bench", "format_table", "write_json"]


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    steps: int
    pd_mean: float
    pd_below_threshold_fraction: float
    exposure_fraction: float
    js_mean: float
    first_warning_step: int | None


@dataclass(frozen=True)
class BenchRow:
    scenario: str
    policy: str
    episodes: int
    reward_mean: float
    reward_std: float
    pd_mean: float
    jammed_fraction: float
    exposure_fraction: float
    js_mean: float
    first_warning_s: float | None


def run_episode(
    scenario: Scenario, policy: Policy, *, seed: int, pd_threshold: float = 0.5
) -> EpisodeResult:
    """跑一局，返回聚合指标。"""
    from .env import SpectrumFogEnv

    env = SpectrumFogEnv(scenario, pd_threshold=pd_threshold)
    observation, _ = env.reset(seed=seed)
    policy.reset(seed=seed)

    total_reward = 0.0
    pd_values: list[float] = []
    js_values: list[float] = []
    jammed = 0
    exposed = 0
    first_warning: int | None = None
    steps = 0

    while True:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        pd_value = float(info.get("pd", 0.0))
        pd_values.append(pd_value)
        js_values.append(float(info.get("js", 0.0)))
        if pd_value < pd_threshold:
            jammed += 1
        if info.get("localized"):
            exposed += 1
            if first_warning is None:
                first_warning = steps

        if terminated or truncated:
            break

    return EpisodeResult(
        reward=total_reward,
        steps=steps,
        pd_mean=sum(pd_values) / max(1, len(pd_values)),
        pd_below_threshold_fraction=jammed / max(1, steps),
        exposure_fraction=exposed / max(1, steps),
        js_mean=sum(js_values) / max(1, len(js_values)),
        first_warning_step=first_warning,
    )


def run_bench(
    scenarios: Sequence[Scenario],
    policy_names: Iterable[str] = ("random", "always-jam", "stealth"),
    *,
    episodes: int = 5,
    seeds: Sequence[int] | None = None,
    pd_threshold: float = 0.5,
) -> list[BenchRow]:
    """对每个 (想定, 策略) 跑若干局，汇总成表。

    ``seeds`` 给定时按它跑（排行榜要固定种子），否则用 ``range(episodes)``。
    """
    seed_list = tuple(seeds) if seeds is not None else tuple(range(episodes))
    if not seed_list:
        raise ValueError("种子列表不能为空")
    if len(seed_list) < 1:
        raise ValueError(f"episodes 至少 1，收到 {episodes}")

    rows: list[BenchRow] = []
    for scenario in scenarios:
        for name in policy_names:
            results = [
                run_episode(
                    scenario,
                    make_policy(name),
                    seed=seed,
                    pd_threshold=pd_threshold,
                )
                for seed in seed_list
            ]
            rewards = [result.reward for result in results]
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
            first_warnings = [
                result.first_warning_step for result in results if result.first_warning_step
            ]
            dt = scenario.meta.dt
            rows.append(
                BenchRow(
                    scenario=scenario.meta.name,
                    policy=name,
                    episodes=len(results),
                    reward_mean=mean_reward,
                    reward_std=variance**0.5,
                    pd_mean=sum(result.pd_mean for result in results) / len(results),
                    jammed_fraction=sum(
                        result.pd_below_threshold_fraction for result in results
                    )
                    / len(results),
                    exposure_fraction=sum(
                        result.exposure_fraction for result in results
                    )
                    / len(results),
                    js_mean=sum(result.js_mean for result in results) / len(results),
                    first_warning_s=(
                        (sum(first_warnings) / len(first_warnings)) * dt
                        if first_warnings
                        else None
                    ),
                )
            )
    rows.sort(key=lambda row: (row.scenario, -row.reward_mean))
    return rows


def format_table(rows: Sequence[BenchRow]) -> str:
    """定宽表格（终端友好，也能直接贴进 markdown 代码块）。"""
    header = (
        f"{'scenario':<14} {'policy':<12} {'reward':>12} {'jammed':>7} "
        f"{'exposed':>8} {'Pd mean':>8} {'J/S mean':>10} {'首告警[s]':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        warning = "—" if row.first_warning_s is None else f"{row.first_warning_s:.0f}"
        lines.append(
            f"{row.scenario:<14} {row.policy:<12} "
            f"{row.reward_mean:>7.1f}±{row.reward_std:>3.0f} "
            f"{row.jammed_fraction * 100:>6.0f}% "
            f"{row.exposure_fraction * 100:>7.0f}% "
            f"{row.pd_mean:>8.3f} {row.js_mean:>10.2f} {warning:>10}"
        )
    return "\n".join(lines)


def write_json(path: str | Path, rows: Sequence[BenchRow]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "spectrumfog.bench/1",
        "rows": [asdict(row) for row in rows],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target
