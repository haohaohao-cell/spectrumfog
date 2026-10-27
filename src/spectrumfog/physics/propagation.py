"""传播模型：自由空间（Friis）+ 路径损耗。

v1 只做自由空间。绕射 / 地形 / 大气吸收放在 T1 之后按需加，
接口先定成"返回线性功率比"，将来换模型不影响调用方。

内部一律 SI：W / m / Hz。
"""

from __future__ import annotations

import numpy as np

from ..units import SPEED_OF_LIGHT
from .radar import _require_positive

__all__ = [
    "free_space_path_loss",
    "free_space_path_loss_db",
    "received_power",
    "received_power_dbw",
]


def free_space_path_loss(freq_hz, range_m):
    """自由空间路径损耗，线性功率比 L = (4*pi*d/lambda)^2。

    注意：这里没有取倒数 —— L 是"损耗因子"，收到的功率 = 发射功率 / L。
    与 §6.1 里雷达方程的 ``L``（系统损耗）不是同一个东西，别混用。
    """
    _require_positive(freq_hz=freq_hz, range_m=range_m)
    lam = SPEED_OF_LIGHT / np.asarray(freq_hz, dtype=float)
    return (4.0 * np.pi * np.asarray(range_m, dtype=float) / lam) ** 2


def free_space_path_loss_db(freq_hz, range_m):
    """自由空间路径损耗，dB（= 10*log10(L)）。"""
    return 10.0 * np.log10(free_space_path_loss(freq_hz=freq_hz, range_m=range_m))


def received_power(ptx_w, gtx, grx, freq_hz, range_m):
    """Friis：Prx = Ptx * Gtx * Grx * (lam / (4*pi*d))^2，单位 W。"""
    _require_positive(
        ptx_w=ptx_w, gtx=gtx, grx=grx, freq_hz=freq_hz, range_m=range_m
    )
    lam = SPEED_OF_LIGHT / np.asarray(freq_hz, dtype=float)
    return (
        np.asarray(ptx_w, dtype=float)
        * np.asarray(gtx, dtype=float)
        * np.asarray(grx, dtype=float)
        * (lam / (4.0 * np.pi * np.asarray(range_m, dtype=float))) ** 2
    )


def received_power_dbw(ptx_w, gtx, grx, freq_hz, range_m):
    """Friis 结果的 dBW 表示（只给 CLI / 日志用）。"""
    prx = received_power(
        ptx_w=ptx_w, gtx=gtx, grx=grx, freq_hz=freq_hz, range_m=range_m
    )
    return 10.0 * np.log10(prx)
