"""传感器模型（v1 只有 ESM；T1 起才真正参与引擎）。

T0 只需要占位 + 校验，让 ``sensors: []`` 这种想定能加载。
雷达传感器 / ESM 的检测流程写在 ``engine/detect.py``（T1）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Sensor"]


class Sensor(BaseModel):
    """一个被动/主动传感器。

    * ``esm``：只收不发，用于测向与识别（v1 的主力）。
    * ``radar``：主动探测，T1 再实现，先留 kind。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: Literal["esm", "radar"] = "esm"
    pos: tuple[float, float] = Field(
        default=(0.0, 0.0), description="[east, north]，单位 m"
    )
    noise_figure_db: float = Field(default=6.0, description="接收机噪声系数 [dB]")
    bw_hz: float = Field(default=1.0e6, gt=0, description="瞬时带宽 [Hz]")

    @property
    def noise_figure(self) -> float:
        """线性噪声系数（物理层用这个）。"""
        return 10.0 ** (self.noise_figure_db / 10.0)
