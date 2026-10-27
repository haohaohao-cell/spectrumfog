"""想定模型层：pydantic 模型 + YAML 加载（单位换算只在这一层发生）。

T0 只落地 ``scenario.py`` / ``platform.py`` / ``emitter.py`` / ``sensor.py``；
移动、波束、诱饵等动态字段随 T1 的引擎一起进来。
"""

from __future__ import annotations

from .emitter import (
    CommsEmitter,
    DecoyEmitter,
    Emitter,
    NoiseJammerEmitter,
    RadarEmitter,
)
from .platform import Platform
from .scenario import (
    Environment,
    Meta,
    Objective,
    Scenario,
    load_scenario,
    normalize_units,
    parse_scenario,
)
from .sensor import Sensor

__all__ = [
    "CommsEmitter",
    "DecoyEmitter",
    "Emitter",
    "NoiseJammerEmitter",
    "RadarEmitter",
    "Platform",
    "Sensor",
    "Meta",
    "Environment",
    "Objective",
    "Scenario",
    "load_scenario",
    "parse_scenario",
    "normalize_units",
]
