"""固定步长时钟。

时间不是浮点累加出来的（那会漂移），而是 ``index * dt`` 直接算出来的整数倍 ——
所以任意 tick 的时间戳在任意机器上都完全一样，replay 才能逐帧对齐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

__all__ = ["Tick", "Clock"]


@dataclass(frozen=True)
class Tick:
    """一次仿真步。"""

    index: int
    t: float  # s
    dt: float  # s


class Clock:
    """t = 0, dt, 2dt, ..., round(duration/dt)·dt，覆盖 [0, duration] 闭区间。

    含端点是为了让动画有"最后一帧"，也让 replay 的最后一行落在 duration 上。
    """

    def __init__(self, dt: float, duration: float) -> None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt 必须为有限正数（秒），收到 {dt!r}")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"duration 必须为有限正数（秒），收到 {duration!r}")
        self.dt = float(dt)
        self.duration = float(duration)
        self.n_ticks = int(round(self.duration / self.dt)) + 1

    @property
    def n_intervals(self) -> int:
        """步数（不含 t=0 那一步）。"""
        return self.n_ticks - 1

    def ticks(self) -> Iterator[Tick]:
        for index in range(self.n_ticks):
            yield Tick(index=index, t=index * self.dt, dt=self.dt)

    def __iter__(self) -> Iterator[Tick]:
        return self.ticks()

    def __len__(self) -> int:
        return self.n_ticks

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Clock(dt={self.dt:g}, duration={self.duration:g}, n_ticks={self.n_ticks})"
