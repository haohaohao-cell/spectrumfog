"""评测与策略层（T3b）。

不是"再写一套物理"：环境里每个 tick 调的都是 ``physics`` 与 ``engine.detect/fusion``
里同一批函数，只是把"谁来决策"交给外部——这正是 §2 决策 3 里"三个面共用一个内核"。

依赖 gymnasium，但只在 ``bench`` extra 里；没装 gymnasium 时 ``spectrumfog bench``
会给出明确提示，其余命令不受影响。
"""

from __future__ import annotations

__all__ = ["SpectrumFogEnv", "POLICIES", "run_bench", "format_table"]


def __getattr__(name: str):  # pragma: no cover - 只是延迟导入，避免硬依赖 gymnasium
    if name == "SpectrumFogEnv":
        from .env import SpectrumFogEnv

        return SpectrumFogEnv
    if name in ("POLICIES", "make_policy"):
        from . import policies

        return getattr(policies, name)
    if name in ("run_bench", "format_table"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
