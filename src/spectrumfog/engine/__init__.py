"""引擎：固定步长时钟 + 确定性主循环 + 事件日志。

设计约束（PROJECT.md §2）：
  * 确定性优先：时间只由 ``dt`` 与 tick 序号决定，随机只来自 seeded RNG。
  * 引擎不碰 UI、不写图；出图在 ``viz``，回放格式在 ``replay.py``。
"""

from __future__ import annotations

from .clock import Clock, Tick
from .sim import Simulation, SimulationResult

__all__ = ["Clock", "Tick", "Simulation", "SimulationResult"]
