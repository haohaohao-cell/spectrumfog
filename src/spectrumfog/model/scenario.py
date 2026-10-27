"""想定模型与 YAML 加载（PROJECT.md §7.1）。

加载流水线只有一条::

    YAML 文本 → yaml.safe_load → normalize_units（dB/km/MHz → SI）
              → pydantic 校验 → Scenario → 物理层

换算只发生在 ``normalize_units`` 这一步，之后内核里全部是 SI（§6.2）。
想定文件里允许的单位后缀：

===========  ============================  ==========================
量            允许写法                      换算成
===========  ============================  ==========================
频率          freq_hz / _khz / _mhz / _ghz  freq_hz [Hz]
带宽          bw_hz / _khz / _mhz / _ghz    bw_hz [Hz]
功率          pt_w / _kw / _mw / _dbm / _dbw  pt_w [W]
干扰功率      pj_w / _kw / _mw / _dbm / _dbw  pj_w [W]
位置          pos（米）/ pos_km / pos_nmi      pos [m]
步长          dt（秒）/ dt_ms                 dt [s]
时长          duration（秒）/ duration_ms      duration [s]
===========  ============================  ==========================

增益 / 损耗 / 噪声系数在 YAML 里写 dB（``gain_db`` 等），由模型层换算成线性值。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..physics.radar import wavelength
from ..units import T0_K, db_to_linear, dbm_to_w, dbw_to_w
from .emitter import NoiseJammerEmitter, RadarEmitter
from .platform import Platform
from .sensor import Sensor

__all__ = [
    "Meta",
    "Environment",
    "Objective",
    "Scenario",
    "load_scenario",
    "parse_scenario",
    "normalize_units",
]


# --- 单位归一化 --------------------------------------------------------------


def _scale(factor: float) -> Callable[[Any], float]:
    return lambda value: float(value) * factor


def _scaled_position(factor: float) -> Callable[[Any], list[float]]:
    return lambda value: [float(component) * factor for component in value]


#: 每个基名 → {单位后缀: 换算函数}
_UNITS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "freq": {
        "hz": _scale(1.0),
        "khz": _scale(1.0e3),
        "mhz": _scale(1.0e6),
        "ghz": _scale(1.0e9),
    },
    "bw": {
        "hz": _scale(1.0),
        "khz": _scale(1.0e3),
        "mhz": _scale(1.0e6),
        "ghz": _scale(1.0e9),
    },
    "pt": {
        "w": _scale(1.0),
        "kw": _scale(1.0e3),
        "mw": _scale(1.0e-3),
        "dbm": dbm_to_w,
        "dbw": dbw_to_w,
    },
    "pj": {
        "w": _scale(1.0),
        "kw": _scale(1.0e3),
        "mw": _scale(1.0e-3),
        "dbm": dbm_to_w,
        "dbw": dbw_to_w,
    },
    "pos": {
        "m": _scaled_position(1.0),
        "km": _scaled_position(1.0e3),
        "nmi": _scaled_position(1852.0),
    },
    "dt": {"s": _scale(1.0), "ms": _scale(1.0e-3)},
    "duration": {"s": _scale(1.0), "ms": _scale(1.0e-3)},
}

#: 基名 → 归一化后的单位后缀（空串表示"不带后缀"，如 pos / dt / duration）
_CANONICAL_UNIT: dict[str, str] = {
    "freq": "hz",
    "bw": "hz",
    "pt": "w",
    "pj": "w",
    "pos": "",
    "dt": "",
    "duration": "",
}


def _match_unit_key(key: str) -> tuple[str, Callable[[Any], Any]] | None:
    """把一个 ``<基名>_<单位>`` 键解析成 (规范化键, 换算函数)；不认识就返回 None。"""
    base, sep, unit = key.rpartition("_")
    if not sep:
        return None
    table = _UNITS.get(base)
    if table is None or unit not in table:
        return None
    canonical_unit = _CANONICAL_UNIT[base]
    canonical_key = f"{base}_{canonical_unit}" if canonical_unit else base
    return canonical_key, table[unit]


def normalize_units(node: Any) -> Any:
    """递归把带单位后缀的键换成 SI 规范键（返回新对象，不改原 dict）。

    同一个量同时给出两种单位（例如 ``freq_hz`` 和 ``freq_mhz``）直接报错，
    而不是让其中一个静默覆盖另一个。
    """
    if isinstance(node, Mapping):
        result: dict[str, Any] = {}
        for key, value in node.items():
            match = _match_unit_key(str(key))
            if match is None:
                result[str(key)] = normalize_units(value)
                continue
            canonical_key, convert = match
            if canonical_key in node and canonical_key != key:
                raise ValueError(
                    f"同一个量给了两种单位：{canonical_key!r} 和 {str(key)!r}，"
                    "请只保留一个"
                )
            result[canonical_key] = convert(normalize_units(value))
        return result
    if isinstance(node, list):
        return [normalize_units(item) for item in node]
    if isinstance(node, tuple):
        return [normalize_units(item) for item in node]
    return node


# --- 模型 --------------------------------------------------------------------


class Meta(BaseModel):
    """想定的元信息；``seed`` 决定 RNG，``dt`` 决定固定步长（§2 决策 1）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    dt: float = Field(default=1.0, gt=0, description="固定步长 [s]")
    duration: float = Field(default=600.0, gt=0, description="仿真时长 [s]")


class Environment(BaseModel):
    """环境参数。``noise_figure_db`` 在 YAML 里写 dB，内部存线性。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    noise_figure: float = Field(
        default=float(db_to_linear(3.0)),
        gt=0,
        validation_alias="noise_figure_db",
        serialization_alias="noise_figure_db",
        description="接收机噪声系数 [线性]，YAML 里写 noise_figure_db",
    )
    temperature_k: float = Field(default=T0_K, gt=0, description="噪声温度 [K]")

    @field_validator("noise_figure", mode="before")
    @classmethod
    def _noise_figure_from_db(cls, value: object) -> float:
        return float(db_to_linear(value))

    @property
    def noise_figure_db(self) -> float:
        """噪声系数的 dB 值（仅展示用）。"""
        return 10.0 * math.log10(self.noise_figure)


class Objective(BaseModel):
    """想定目标（§8 的指标验收条件）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["localize", "detect", "protect_link"]
    target: str = Field(min_length=1, description="目标平台 id")
    cep_m: float | None = Field(default=None, gt=0, description="定位精度门槛 [m]")


class Scenario(BaseModel):
    """一份完整想定。构造后不可变（frozen），保证 replay 可复现。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    meta: Meta
    environment: Environment = Field(default_factory=Environment)
    platforms: tuple[Platform, ...]
    sensors: tuple[Sensor, ...] = ()
    objectives: tuple[Objective, ...] = ()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Scenario":
        if not self.platforms:
            raise ValueError("想定至少需要一个平台")
        platform_ids = [item.id for item in self.platforms]
        duplicates = _duplicates(platform_ids)
        if duplicates:
            raise ValueError(f"平台 id 重复：{duplicates}")
        sensor_ids = [item.id for item in self.sensors]
        duplicates = _duplicates(sensor_ids)
        if duplicates:
            raise ValueError(f"传感器 id 重复：{duplicates}")
        unknown = [
            objective.target
            for objective in self.objectives
            if objective.target not in platform_ids
        ]
        if unknown:
            raise ValueError(f"objectives 指向了不存在的平台：{unknown}")
        return self

    # --- 查询 -------------------------------------------------------------

    def platform(self, platform_id: str) -> Platform:
        """按 id 取平台；不存在抛 ValueError（带可用 id 列表）。"""
        for item in self.platforms:
            if item.id == platform_id:
                return item
        available = ", ".join(item.id for item in self.platforms)
        raise ValueError(f"想定里没有平台 {platform_id!r}；可用：{available}")

    def radars(self) -> list[tuple[Platform, RadarEmitter]]:
        """所有 (平台, 雷达) 组合。"""
        return [
            (platform, emitter)
            for platform in self.platforms
            for emitter in platform.emitters
            if isinstance(emitter, RadarEmitter)
        ]

    def jammers(self) -> list[tuple[Platform, NoiseJammerEmitter]]:
        """所有 (平台, 干扰机) 组合。"""
        return [
            (platform, emitter)
            for platform in self.platforms
            for emitter in platform.emitters
            if isinstance(emitter, NoiseJammerEmitter)
        ]

    def primary_radar(
        self, radar_id: str | None = None
    ) -> tuple[Platform, RadarEmitter]:
        """取雷达：给了 id 就按平台 id 找，否则取第一个。"""
        if radar_id is not None:
            platform = self.platform(radar_id)
            emitter = platform.emitter("radar")
            if not isinstance(emitter, RadarEmitter):
                raise ValueError(f"平台 {radar_id!r} 上没有 radar 辐射源")
            return platform, emitter
        candidates = self.radars()
        if not candidates:
            raise ValueError("想定里没有任何 radar 辐射源")
        return candidates[0]

    def primary_jammer(
        self, jammer_id: str | None = None
    ) -> tuple[Platform, NoiseJammerEmitter]:
        """取干扰机：给了 id 就按平台 id 找，否则取第一个。"""
        if jammer_id is not None:
            platform = self.platform(jammer_id)
            emitter = platform.emitter("noise_jammer")
            if not isinstance(emitter, NoiseJammerEmitter):
                raise ValueError(f"平台 {jammer_id!r} 上没有 noise_jammer 辐射源")
            return platform, emitter
        candidates = self.jammers()
        if not candidates:
            raise ValueError("想定里没有任何 noise_jammer 辐射源")
        return candidates[0]

    def link_params(
        self, radar_id: str | None = None, jammer_id: str | None = None
    ) -> dict[str, float]:
        """一把雷达 + 一台干扰机的自屏蔽链路参数，可直接喂给 ``physics.radar``。

        返回的键与 ``js_ratio_self_screen`` / ``pd_from_powers`` 的入参同名；
        ``burn_through_range`` 不接受 ``lam``（波长相消），用的时候去掉即可。
        """
        _, radar_emitter = self.primary_radar(radar_id)
        _, jammer_emitter = self.primary_jammer(jammer_id)
        return {
            "pt_w": float(radar_emitter.pt_w),
            "gain": float(radar_emitter.gain),
            "sigma_m2": float(radar_emitter.sigma_m2),
            "lam": float(wavelength(radar_emitter.freq_hz)),
            "pj_w": float(jammer_emitter.pj_w),
            "gj": float(jammer_emitter.gain),
            "bw_r_hz": float(radar_emitter.bw_hz),
            "bw_j_hz": float(jammer_emitter.bw_hz),
            "loss": float(radar_emitter.loss),
        }

    def standoff_m(
        self, radar_id: str | None = None, jammer_id: str | None = None
    ) -> float:
        """雷达平台与干扰平台之间的平面距离 [m]。"""
        radar_platform, _ = self.primary_radar(radar_id)
        jammer_platform, _ = self.primary_jammer(jammer_id)
        return radar_platform.distance_to(jammer_platform)


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


# --- 加载入口 ----------------------------------------------------------------


def parse_scenario(data: Mapping[str, Any]) -> Scenario:
    """从已解析的 dict 构造想定（会先做单位归一化）。"""
    if not isinstance(data, Mapping):
        raise ValueError(f"想定顶层必须是映射（mapping），收到 {type(data).__name__}")
    return Scenario.model_validate(normalize_units(dict(data)))


def load_scenario(path: str | Path) -> Scenario:
    """从 YAML 文件加载想定。"""
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"想定文件不存在：{scenario_path}")
    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # 语法错就明确说是 YAML 错，别混进 pydantic 报错里
        raise ValueError(f"YAML 解析失败：{scenario_path}\n{exc}") from exc
    if raw is None:
        raise ValueError(f"想定文件是空的：{scenario_path}")
    try:
        return parse_scenario(raw)
    except ValueError as exc:
        raise ValueError(f"想定校验失败：{scenario_path}\n{exc}") from exc
