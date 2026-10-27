"""三个 baseline 策略。

**只吃观测**，不看 ``info``（``info`` 里有 Pd、CEP 这类上帝视角的数字）。
不然 benchmark 就成了"用未来数据作弊"，表格也不可比。
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .env import DIRECTIONS, POWER_LEVELS

__all__ = [
    "Policy",
    "RandomPolicy",
    "AlwaysJamPolicy",
    "StealthPolicy",
    "POLICIES",
    "make_policy",
]

_ACTION_COUNT = len(DIRECTIONS) * len(POWER_LEVELS)
_STAY, _PLUS_X, _MINUS_X, _PLUS_Y, _MINUS_Y = range(len(DIRECTIONS))
_OFF, _HALF, _FULL = range(len(POWER_LEVELS))


class Policy(Protocol):
    """策略接口：观测进，动作出。"""

    name: str

    def reset(self, *, seed: int | None = None) -> None: ...

    def act(self, observation: np.ndarray) -> int: ...


class RandomPolicy:
    """均匀随机动作——用来给表格一个下限。"""

    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def act(self, observation: np.ndarray) -> int:
        return int(self._rng.integers(0, _ACTION_COUNT))


class AlwaysJamPolicy:
    """原地满功率——压制最强、也最容易被定位。"""

    name = "always-jam"

    def reset(self, *, seed: int | None = None) -> None:
        return None

    def act(self, observation: np.ndarray) -> int:
        return _STAY * len(POWER_LEVELS) + _FULL


class StealthPolicy:
    """"静默接近 → 贴脸开火 → 收到告警就撤"的手写策略。

    这条策略专门利用这个环境的核心权衡：**开机才有压制，开机就会被截获**。
    所以它在远处保持静默（只动不开），压到相对距离 25% 以内才开满功率；
    一旦收到告警就关机后撤几帧，再重新靠近。

    只看观测（相对距离、告警位），不读 env 私有状态。
    """

    name = "stealth"

    def __init__(self) -> None:
        self._cooldown = 0

    def reset(self, *, seed: int | None = None) -> None:
        self._cooldown = 0

    def act(self, observation: np.ndarray) -> int:
        x = float(observation[0])
        relative_range = float(observation[4])
        warned = float(observation[5]) > 0.5

        if warned:
            self._cooldown = 5
        if self._cooldown > 0:
            self._cooldown -= 1
            away = _PLUS_X if x >= 0.0 else _MINUS_X
            return away * len(POWER_LEVELS) + _OFF
        if relative_range > 0.25:
            toward = _MINUS_X if x >= 0.0 else _PLUS_X
            return toward * len(POWER_LEVELS) + _OFF  # 静默接近
        return _STAY * len(POWER_LEVELS) + _FULL  # 压到近处再开火


POLICIES: tuple[type, ...] = (RandomPolicy, AlwaysJamPolicy, StealthPolicy)


def make_policy(name: str) -> Policy:
    """按名字取策略工厂。"""
    lookup = {policy_cls.name: policy_cls for policy_cls in POLICIES}
    if name not in lookup:
        available = ", ".join(sorted(lookup))
        raise ValueError(f"未知策略 {name!r}；可用：{available}")
    return lookup[name]()
