"""平台（平台 = 一个在 2D 平面上的实体，挂若干辐射源与传感器）。

坐标约定：**米**，ENU 平面上的 ``[east, north]``。
v1 不做地形与高度；3D 是明确非目标（§3）。

运动模型（T1）：分段线性航路点。纯确定性的时间函数，不碰 RNG ——
同一时刻同一平台永远在同一个位置，这是 replay 可逐帧复现的前提（§2 决策 1）。
"""

from __future__ import annotations

import bisect
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .emitter import Emitter

__all__ = ["Waypoint", "Platform"]


class Waypoint(BaseModel):
    """航路点：``t`` 时刻在 ``pos``。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t: float = Field(ge=0.0, description="时间 [s]")
    pos: tuple[float, float] = Field(description="[east, north]，单位 m")


class Platform(BaseModel):
    """想定里的一个平台。

    ``pos`` 是初始位置（也是没有航路点时的唯一位置），单位米。
    ``emitters`` 为空表示"静默平台"——v1 允许，用来放只收不发的传感器或纯目标。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    side: Literal["blue", "red"]
    pos: tuple[float, float] = Field(
        default=(0.0, 0.0), description="[east, north]，单位 m"
    )
    waypoints: tuple[Waypoint, ...] = Field(
        default=(), description="航路点，按时间排序后使用；空 = 静止"
    )
    emitters: tuple[Emitter, ...] = ()

    @model_validator(mode="after")
    def _sort_waypoints(self) -> "Platform":
        if not self.waypoints:
            return self
        ordered = tuple(sorted(self.waypoints, key=lambda waypoint: waypoint.t))
        times = [waypoint.t for waypoint in ordered]
        if len(set(times)) != len(times):
            raise ValueError(f"平台 {self.id!r} 的航路点有重复时刻：{times}")
        object.__setattr__(self, "waypoints", ordered)
        return self

    # --- 查询 -------------------------------------------------------------

    def distance_to(self, other: "Platform") -> float:
        """两平台在 t = 0 时的平面距离 [m]。"""
        return math.dist(self.pos, other.pos)

    def emitter(self, kind: str) -> Emitter | None:
        """按 kind 取第一个辐射源（v1 每个平台通常只挂一个）。"""
        for item in self.emitters:
            if item.kind == kind:
                return item
        return None

    # --- 运动 -------------------------------------------------------------

    def position_at(self, t_s: float) -> tuple[float, float]:
        """``t_s`` 时刻的位置 [m]，分段线性插值。

        * 没有航路点 → 原地不动（返回 ``pos``）。
        * 早于第一个航路点 → 停在第一个点；晚于最后一个 → 停在最后一个点
          （不做外推：外推会让 replay 里的位置依赖仿真时长）。
        """
        if not self.waypoints:
            return self.pos

        times = [waypoint.t for waypoint in self.waypoints]
        if t_s <= times[0]:
            return self.waypoints[0].pos
        if t_s >= times[-1]:
            return self.waypoints[-1].pos

        index = bisect.bisect_right(times, t_s) - 1
        before = self.waypoints[index]
        after = self.waypoints[index + 1]
        span = after.t - before.t
        ratio = (t_s - before.t) / span
        return (
            before.pos[0] + (after.pos[0] - before.pos[0]) * ratio,
            before.pos[1] + (after.pos[1] - before.pos[1]) * ratio,
        )
