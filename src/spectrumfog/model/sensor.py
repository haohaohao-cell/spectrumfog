"""传感器模型（v1 只有 ESM；T3a 起真正参与观测与融合）。

ESM 只收不发：截获敌方辐射源的频率、到达角（AOA）、到达时间（TOA）与功率，
再交给 ``engine/fusion.py`` 做定位。检测流程在 ``engine/detect.py``。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..units import db_to_linear

__all__ = ["Sensor"]


class Sensor(BaseModel):
    """一个被动传感器。

    ``side`` 决定它观察谁：ESM 只看对方阵营的辐射源。
    精度量级（``pfa_scan``、噪声系数、带宽）都是虚构值，用途是让"信号越弱、测向越糊"
    这条关系在仿真里成立。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: Literal["esm", "radar"] = "esm"
    side: Literal["blue", "red"] = "blue"
    pos: tuple[float, float] = Field(
        default=(0.0, 0.0), description="[east, north]，单位 m"
    )
    noise_figure_db: float = Field(default=6.0, description="接收机噪声系数 [dB]")
    bw_hz: float = Field(default=1.0e6, gt=0, description="瞬时带宽 [Hz]")
    gain_db: float = Field(default=0.0, description="天线增益 [dBi]（v1 按全向处理）")
    pfa_scan: float = Field(
        default=1.0e-4,
        gt=0.0,
        lt=1.0,
        description="单次扫描的虚警概率（虚假目标的来源）",
    )

    @property
    def noise_figure(self) -> float:
        """线性噪声系数（物理层用这个）。"""
        return 10.0 ** (self.noise_figure_db / 10.0)

    @property
    def gain(self) -> float:
        """线性天线增益。"""
        return float(db_to_linear(self.gain_db))
