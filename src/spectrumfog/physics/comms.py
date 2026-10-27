"""通信链路预算、Eb/N0 与抗干扰余量。

对应 PROJECT.md §6.1 的链路预算三条公式::

    Prx = Ptx * Gtx * Grx * (lam / (4*pi*d))**2
    EbN0 = Prx / (N0 * Rb)
    AJ_margin_dB = Gp + Mc - required_JS_dB

v1 只做标量/数组的纯计算，调制与编码的具体增益由想定给定（参数全虚构）。
内部一律 SI：W / m / Hz / s。
"""

from __future__ import annotations

import numpy as np

from ..units import K_BOLTZMANN, T0_K
from .propagation import received_power as _friis
from .radar import _require_positive

__all__ = [
    "noise_psd",
    "link_received_power",
    "ebn0",
    "ebn0_db",
    "aj_margin_db",
    "link_available",
]


def noise_psd(noise_figure=1.0, temperature_k=T0_K):
    """噪声功率谱密度 N0 = k * T * F，单位 W/Hz。F 为线性噪声系数。"""
    _require_positive(noise_figure=noise_figure, temperature_k=temperature_k)
    return (
        K_BOLTZMANN
        * np.asarray(temperature_k, dtype=float)
        * np.asarray(noise_figure, dtype=float)
    )


def link_received_power(ptx_w, gtx, grx, freq_hz, range_m):
    """接收功率 Prx = Ptx*Gtx*Grx*(lam/(4*pi*d))^2，单位 W。

    与 ``propagation.received_power`` 是同一个式子 —— 这里只是给链路预算
    一个语义化入口，避免 comms 的调用方直接依赖 propagation 的命名。
    """
    return _friis(
        ptx_w=ptx_w, gtx=gtx, grx=grx, freq_hz=freq_hz, range_m=range_m
    )


def ebn0(prx_w, noise_psd_w_per_hz, bit_rate_bps):
    """Eb/N0 = Prx / (N0 * Rb)，返回线性值。"""
    _require_positive(
        prx_w=prx_w, noise_psd_w_per_hz=noise_psd_w_per_hz, bit_rate_bps=bit_rate_bps
    )
    return np.asarray(prx_w, dtype=float) / (
        np.asarray(noise_psd_w_per_hz, dtype=float)
        * np.asarray(bit_rate_bps, dtype=float)
    )


def ebn0_db(prx_w, noise_psd_w_per_hz, bit_rate_bps):
    """Eb/N0 的 dB 值（只给 CLI / 日志用）。"""
    return 10.0 * np.log10(
        ebn0(
            prx_w=prx_w,
            noise_psd_w_per_hz=noise_psd_w_per_hz,
            bit_rate_bps=bit_rate_bps,
        )
    )


def aj_margin_db(processing_gain_db, coding_gain_db=0.0, required_js_db=0.0):
    """抗干扰余量 AJ_margin = Gp + Mc - required_JS，单位 dB。

    > 0 视为链路可用（见 ``link_available``）。三个参数都是 dB 值，dB 域里
    加减即乘除，所以这里不做任何线性换算；传 dB 而不是线性增益是签名层面的约定。
    """
    return (
        np.asarray(processing_gain_db, dtype=float)
        + np.asarray(coding_gain_db, dtype=float)
        - np.asarray(required_js_db, dtype=float)
    )


def link_available(processing_gain_db, coding_gain_db=0.0, required_js_db=0.0):
    """链路可用判定：AJ_margin > 0（对应指标 Link_avail，§8）。"""
    return aj_margin_db(processing_gain_db, coding_gain_db, required_js_db) > 0.0
