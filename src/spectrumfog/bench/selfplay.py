"""自训练：线性 softmax 策略 + 交叉熵方法（CEM），纯 numpy。

为什么不用强化学习库：这个项目的依赖里没有 torch，也不想为了一个"两种动作空间、
300 步一局"的小环境去拖一个深度学习栈。CEM 在这里够用——它的思想就是
"采样一群策略 → 跑分 → 留下最好的那几个 → 收紧分布"，十几行能写完，也不藏东西。

**自训练（self-play）** 的做法是一替一替地练：

    第 g 代：红方对着"上一代的蓝方"练；蓝方再对着"刚练好的红方"练

两边都在变强，所以绝对分数没有意义，有意义的是**相对上一代的对手**。

已知局限（写在这里，别当成没看见）：

* 每代样本很少（几个候选 × 一两局），训练信号很弱；
* 策略是线性的，表达力有限；
* 没有价值函数、没有折扣回报，只优化"一局的总分"。

这是"机制验证"——证明环境能训、能看到行为变化，不是"调出一个强策略"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..model.scenario import Scenario
from .duel import DuelEnv

__all__ = [
    "LinearSoftmax",
    "TrainingHistory",
    "self_play",
    "format_history",
]


def features(observation: np.ndarray, *, periods: int = 4) -> np.ndarray:
    """把 7 维观测扩成带时间基函数的向量。

    线性策略没有记忆，只看得见当帧的 ``t_frac``；要让它表达"每 5 帧开一次"，
    必须给它 sin/cos 基函数——这是"策略表达力"问题，不是训练问题。
    """
    observation = np.asarray(observation, dtype=float)
    t_frac = float(observation[3])
    extra = []
    for harmonic in range(1, periods + 1):
        angle = 2.0 * np.pi * harmonic * t_frac
        extra.extend([np.sin(angle), np.cos(angle)])
    return np.concatenate([observation, np.asarray(extra, dtype=float)])


class LinearSoftmax:
    """7 维观测 → 动作分布的线性策略。参数是一个 (n_actions, n_obs) 矩阵。"""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    @classmethod
    def random(cls, n_actions: int, n_obs: int, rng: np.random.Generator, scale: float = 0.1):
        return cls(rng.normal(0.0, scale, size=(n_actions, n_obs)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    def logits(self, observation: np.ndarray) -> np.ndarray:
        return self.weights @ np.asarray(observation, dtype=float)

    def act(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        use_features: bool = False,
    ) -> int:
        """给了 rng 就按分布采样（训练），否则取最大（评估）。"""
        vector = features(observation) if use_features else observation
        logits = self.logits(vector)
        if rng is None:
            return int(np.argmax(logits))
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return int(rng.choice(len(probabilities), p=probabilities))


@dataclass
class TrainingHistory:
    """每一代的成绩。分数是"对上一代对手"的，不是绝对分。"""

    generations: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.generations.append(row)

    def __len__(self) -> int:
        return len(self.generations)


def _episode(
    scenario: Scenario,
    red: LinearSoftmax,
    blue: LinearSoftmax,
    *,
    seed: int,
    use_features: bool = False,
) -> tuple[float, float, float]:
    """跑一局，返回 (红方得分, 蓝方得分, 蓝方开机占比)。"""
    env = DuelEnv(scenario)
    observations, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    red_obs, blue_obs = observations["red"], observations["blue"]
    red_total = 0.0
    blue_total = 0.0
    on_ticks = 0
    steps = 0
    while True:
        red_action = red.act(red_obs, rng=rng, use_features=use_features)
        blue_action = blue.act(blue_obs, rng=rng, use_features=use_features)
        observations, rewards, terminated, truncated, info = env.step(red_action, blue_action)
        red_obs, blue_obs = observations["red"], observations["blue"]
        red_total += rewards["red"]
        blue_total += rewards["blue"]
        on_ticks += int(info["radiating"])
        steps += 1
        if terminated or truncated:
            break
    return red_total, blue_total, on_ticks / max(1, steps)


def _score(
    scenario: Scenario,
    candidate: LinearSoftmax,
    opponent: LinearSoftmax,
    side: str,
    *,
    seeds: Sequence[int],
    use_features: bool = False,
) -> float:
    total = 0.0
    for seed in seeds:
        red = candidate if side == "red" else opponent
        blue = candidate if side == "blue" else opponent
        red_score, blue_score, _ = _episode(
            scenario, red, blue, seed=seed, use_features=use_features
        )
        total += red_score if side == "red" else blue_score
    return total / len(seeds)


def _cem(
    scenario: Scenario,
    side: str,
    opponent: LinearSoftmax,
    initial: LinearSoftmax,
    *,
    iterations: int,
    population: int,
    elite_fraction: float = 0.3,
    seeds: Sequence[int] = (0, 1),
    seed: int = 0,
    use_features: bool = False,
) -> LinearSoftmax:
    """在给定对手下，用 CEM 把一侧练几轮。"""
    rng = np.random.default_rng(seed)
    mean = initial.weights.copy()
    std = np.full_like(mean, 0.5)
    n_elite = max(1, int(round(population * elite_fraction)))

    for _ in range(iterations):
        candidates = [LinearSoftmax(mean + rng.normal(0.0, std)) for _ in range(population)]
        scored = [
            (
                _score(
                    scenario,
                    candidate,
                    opponent,
                    side,
                    seeds=seeds,
                    use_features=use_features,
                ),
                candidate,
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.stack([candidate.weights for _, candidate in scored[:n_elite]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3

    return LinearSoftmax(mean)


def self_play(
    scenario: Scenario,
    *,
    generations: int = 4,
    iterations: int = 4,
    population: int = 4,
    seeds: Sequence[int] = (0,),
    seed: int = 0,
    use_features: bool = False,
) -> tuple[TrainingHistory, LinearSoftmax, LinearSoftmax]:
    """红蓝交替训练，返回历史与双方最终策略。"""
    rng = np.random.default_rng(seed)
    n_obs = features(np.zeros(7)).size if use_features else 7
    red = LinearSoftmax.random(15, n_obs, rng)
    blue = LinearSoftmax.random(3, n_obs, rng)
    history = TrainingHistory()

    # 第 0 代：两个随机策略互相打，作为基线
    red_score, blue_score, on_fraction = _episode(
        scenario, red, blue, seed=seeds[0], use_features=use_features
    )
    history.add(generation=0, red=red_score, blue=blue_score, blue_on=on_fraction)

    for generation in range(1, generations + 1):
        red = _cem(
            scenario,
            "red",
            blue,
            red,
            iterations=iterations,
            population=population,
            seeds=seeds,
            seed=seed + generation * 2,
            use_features=use_features,
        )
        blue = _cem(
            scenario,
            "blue",
            red,
            blue,
            iterations=iterations,
            population=population,
            seeds=seeds,
            seed=seed + generation * 2 + 1,
            use_features=use_features,
        )
        red_score, blue_score, on_fraction = _episode(
            scenario, red, blue, seed=seeds[0], use_features=use_features
        )
        history.add(
            generation=generation,
            red=red_score,
            blue=blue_score,
            blue_on=on_fraction,
        )

    return history, red, blue


def format_history(history: TrainingHistory) -> str:
    lines = [
        f"{'代':>3} {'红方':>9} {'蓝方':>9} {'蓝方开机':>8}",
        "-" * 32,
    ]
    for row in history.generations:
        lines.append(
            f"{row['generation']:>3} {row['red']:>9.1f} {row['blue']:>9.1f} "
            f"{row['blue_on'] * 100:>7.0f}%"
        )
    return "\n".join(lines)


def _demo() -> None:  # pragma: no cover - 手动运行用
    from pathlib import Path

    from ..model.scenario import load_scenario

    scenario = load_scenario(Path("scenarios") / "08_duel_2p.yaml")
    history, red, blue = self_play(scenario, generations=3, iterations=3, population=4)
    print("SpectrumFog · 自训练（红蓝交替）")
    print(format_history(history))
    print()
    print("蓝方最终策略的动作偏好：", np.round(np.abs(blue.weights).sum(axis=1), 2))


if __name__ == "__main__":  # pragma: no cover
    _demo()
