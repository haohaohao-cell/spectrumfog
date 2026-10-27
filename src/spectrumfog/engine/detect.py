"""ESM 观测模型：概率化测量、漏检、虚假目标（PROJECT.md §2 决策 5）。

对每一条 (传感器, 敌方辐射源) 组合、每个 tick：

1. 单程 Friis 算到达功率 → 传感器信噪比（传感器天线按全向处理）；
2. 用与雷达相同的 Swerling I 闭式算单次扫描的 Pd；
3. 掷点决定这一帧是否检出；
4. 检出则生成带噪声的 AOA / TOA / 频率 / 功率；
5. 按虚假目标率生成无源量测（``emitter=None``），用于喂给融合制造假轨迹。

噪声量级都是**虚构值**，但缩放关系写死在这里：AOA 精度 ∝ 1/√SNR，
所以"信号越弱、测向越糊"是模型内生的，不是后期硬调的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..physics import radar
from ..units import SPEED_OF_LIGHT, T0_K

__all__ = ["EsmMeasurement", "observe", "false_alarm", "aoa_sigma"]


@dataclass(frozen=True)
class EsmMeasurement:
    """一条 ESM 量测（或一次未检出）。"""

    sensor_id: str
    sensor_pos: tuple[float, float]
    emitter_id: str | None  # 虚假目标时是 None
    t_s: float
    pd: float
    draw: float
    detected: bool
    aoa_rad: float | None = None
    sigma_aoa_rad: float | None = None
    toa_s: float | None = None
    sigma_toa_s: float | None = None
    freq_hz: float | None = None
    power_dbm: float | None = None

    def to_event(self) -> dict:
        """转成 replay 里的一行（自包含，不依赖其他行）。"""
        return {
            "kind": "measurement",
            "t": self.t_s,
            "sensor": self.sensor_id,
            "sensor_pos": [float(self.sensor_pos[0]), float(self.sensor_pos[1])],
            "emitter": self.emitter_id,
            "detected": self.detected,
            "pd": self.pd,
            "draw": self.draw,
            "aoa_rad": self.aoa_rad,
            "sigma_aoa_rad": self.sigma_aoa_rad,
            "toa_s": self.toa_s,
            "sigma_toa_s": self.sigma_toa_s,
            "freq_hz": self.freq_hz,
            "power_dbm": self.power_dbm,
        }


#: AOA 精度的基准：在参考信噪比下 2°；σ ∝ 1/√SNR
_REFERENCE_SNR = 100.0
_BASE_SIGMA_AOA_RAD = math.radians(2.0)
_SIGMA_TOA_S = 1.0e-7  # 30 m，虚构值
_SIGMA_FREQ_HZ = 1.0e3
_SIGMA_POWER_DB = 1.0

#: 最低 / 最高测向精度（防止给一个"完美"或"完全没用"的方差）
_MIN_SIGMA_AOA_RAD = math.radians(0.2)
_MAX_SIGMA_AOA_RAD = math.radians(20.0)


def aoa_sigma(snr: float) -> float:
    """测向精度随信噪比变化：σ = 2° · √(100/SNR)，夹在 [0.2°, 20°]。"""
    if not math.isfinite(snr) or snr <= 0.0:
        return _MAX_SIGMA_AOA_RAD
    sigma = _BASE_SIGMA_AOA_RAD * math.sqrt(_REFERENCE_SNR / snr)
    return float(np.clip(sigma, _MIN_SIGMA_AOA_RAD, _MAX_SIGMA_AOA_RAD))


def observe(
    *,
    sensor_id: str,
    sensor_pos: tuple[float, float],
    sensor_bw_hz: float,
    sensor_noise_figure: float,
    sensor_gain: float = 1.0,
    emitter_id: str,
    emitter_pos: tuple[float, float],
    emitter_power_w: float,
    emitter_gain: float,
    emitter_freq_hz: float,
    t_s: float,
    pfa: float,
    rng: np.random.Generator,
) -> EsmMeasurement:
    """一次 ESM 扫描。返回量测（含未检出），调用方决定要不要记录。"""
    range_m = float(math.dist(sensor_pos, emitter_pos))
    wavelength = SPEED_OF_LIGHT / emitter_freq_hz
    # 单程 Friis：Prx = Ptx·Gtx·Grx·(λ/4πR)²
    prx_w = (
        emitter_power_w
        * emitter_gain
        * sensor_gain
        * (wavelength / (4.0 * math.pi * max(range_m, 1.0))) ** 2
    )
    pn_w = float(
        radar.noise_power(
            bw_hz=sensor_bw_hz,
            noise_figure=sensor_noise_figure,
            temperature_k=T0_K,
        )
    )
    snr = prx_w / pn_w
    pd = float(radar.pd_swerling1(pfa, snr))
    draw = float(rng.random())
    detected = draw < pd

    if not detected:
        return EsmMeasurement(
            sensor_id=sensor_id,
            sensor_pos=sensor_pos,
            emitter_id=emitter_id,
            t_s=t_s,
            pd=pd,
            draw=draw,
            detected=False,
        )

    sigma_aoa = aoa_sigma(snr)
    true_bearing = math.atan2(
        emitter_pos[1] - sensor_pos[1], emitter_pos[0] - sensor_pos[0]
    )
    aoa = _wrap(true_bearing + float(rng.normal(0.0, sigma_aoa)))
    toa = t_s + range_m / SPEED_OF_LIGHT + float(rng.normal(0.0, _SIGMA_TOA_S))

    return EsmMeasurement(
        sensor_id=sensor_id,
        sensor_pos=sensor_pos,
        emitter_id=emitter_id,
        t_s=t_s,
        pd=pd,
        draw=draw,
        detected=True,
        aoa_rad=aoa,
        sigma_aoa_rad=sigma_aoa,
        toa_s=toa,
        sigma_toa_s=_SIGMA_TOA_S,
        freq_hz=emitter_freq_hz + float(rng.normal(0.0, _SIGMA_FREQ_HZ)),
        power_dbm=10.0 * math.log10(max(prx_w, 1e-30) * 1e3)
        + float(rng.normal(0.0, _SIGMA_POWER_DB)),
    )


def false_alarm(
    *,
    sensor_id: str,
    sensor_pos: tuple[float, float],
    t_s: float,
    pfa_scan: float,
    rng: np.random.Generator,
) -> EsmMeasurement | None:
    """虚假目标：按给定概率产生一条无源量测（方位均匀、功率贴近噪声）。"""
    if pfa_scan <= 0.0:
        return None
    draw = float(rng.random())
    if draw >= pfa_scan:
        return None
    aoa = float(rng.uniform(-math.pi, math.pi))
    return EsmMeasurement(
        sensor_id=sensor_id,
        sensor_pos=sensor_pos,
        emitter_id=None,
        t_s=t_s,
        pd=0.0,
        draw=draw,
        detected=True,
        aoa_rad=aoa,
        sigma_aoa_rad=_BASE_SIGMA_AOA_RAD,
        toa_s=t_s + float(rng.uniform(0.0, 1.0e-4)),
        sigma_toa_s=_SIGMA_TOA_S,
        freq_hz=None,
        power_dbm=float(rng.normal(-110.0, 3.0)),
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
