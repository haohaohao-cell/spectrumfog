"""辐射源模型：想定里的"谁在发什么"。

单位策略（PROJECT.md §6.2）
--------------------------
* 功率 / 带宽一律 **SI**（W、Hz）。
* 增益与损耗在 YAML 里写 **dB**（``gain_db`` / ``loss_db``），加载时立刻换算：
  模型字段 ``gain`` / ``loss`` 存的已经是**线性**值，物理层直接拿去用，
  不会出现"这个 gain 是 dB 还是线性"的歧义。
* 写出时 ``model_dump(by_alias=True)`` 会还原成 ``gain_db``，便于回写日志。

参数全虚构（§12 合规红线），不含任何真实装备性能数据。
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..units import db_to_linear

__all__ = ["RadarEmitter", "NoiseJammerEmitter", "Emitter"]


class _EmitterBase(BaseModel):
    """所有辐射源的公共字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    freq_hz: float = Field(gt=0, description="中心频率 [Hz]")
    bw_hz: float = Field(gt=0, description="发射/接收带宽 [Hz]")


class RadarEmitter(_EmitterBase):
    """雷达：单站，收发共用一副天线（对应 §6.1 的 G**2）。"""

    kind: Literal["radar"]
    pt_w: float = Field(gt=0, description="峰值发射功率 [W]")
    gain: float = Field(
        gt=0,
        validation_alias="gain_db",
        serialization_alias="gain_db",
        description="天线增益 [线性]，YAML 里写 gain_db",
    )
    sigma_m2: float = Field(gt=0, description="目标 RCS [m^2]")
    loss: float = Field(
        default=1.0,
        gt=0,
        validation_alias="loss_db",
        serialization_alias="loss_db",
        description="系统损耗 [线性]，YAML 里写 loss_db",
    )

    @field_validator("gain", mode="before")
    @classmethod
    def _gain_from_db(cls, value: object) -> float:
        return float(db_to_linear(value))

    @field_validator("loss", mode="before")
    @classmethod
    def _loss_from_db(cls, value: object) -> float:
        return float(db_to_linear(value))

    @property
    def gain_db(self) -> float:
        """增益的 dB 值（只用于展示 / 日志；物理层用 ``gain``）。"""
        return 10.0 * math.log10(self.gain)

    @property
    def loss_db(self) -> float:
        """损耗的 dB 值（只用于展示 / 日志；物理层用 ``loss``）。"""
        return 10.0 * math.log10(self.loss)


class NoiseJammerEmitter(_EmitterBase):
    """噪声压制干扰机：把功率摊在 ``bw_hz`` 上，只有落进雷达带宽的部分算数。"""

    kind: Literal["noise_jammer"]
    pj_w: float = Field(gt=0, description="干扰发射功率 [W]")
    gain: float = Field(
        gt=0,
        validation_alias="gain_db",
        serialization_alias="gain_db",
        description="干扰天线在雷达方向上的增益 [线性]，YAML 里写 gain_db",
    )

    @field_validator("gain", mode="before")
    @classmethod
    def _gain_from_db(cls, value: object) -> float:
        return float(db_to_linear(value))

    @property
    def gain_db(self) -> float:
        """增益的 dB 值（只用于展示 / 日志；物理层用 ``gain``）。"""
        return 10.0 * math.log10(self.gain)


#: 想定 emitters 列表的元素类型，按 ``kind`` 字段分派。
Emitter = Annotated[
    Union[RadarEmitter, NoiseJammerEmitter], Field(discriminator="kind")
]
