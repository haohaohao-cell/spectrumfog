"""SI 常量与单位换算。

PROJECT.md §6.2：内部一律 SI（W / m / Hz / s / rad）。
dB / km / MHz 只允许出现在 YAML 输入、CLI 输出和测试边界上，
进内核前立刻用本模块换算成线性值 / 基本单位。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SPEED_OF_LIGHT",
    "K_BOLTZMANN",
    "T0_K",
    "db_to_linear",
    "linear_to_db",
    "dbm_to_w",
    "dbw_to_w",
    "w_to_dbm",
    "km_to_m",
    "nmi_to_m",
    "mhz_to_hz",
    "ghz_to_hz",
    "hz_to_mhz",
    "ms_to_s",
    "deg_to_rad",
    "rad_to_deg",
]

# --- 物理常量（CODATA 2018 / SI 2019 定义值） --------------------------------

SPEED_OF_LIGHT: float = 299_792_458.0  # m/s，真空光速（精确值）
K_BOLTZMANN: float = 1.380649e-23  # J/K，玻尔兹曼常数（定义值）
T0_K: float = 290.0  # K，标准噪声温度（PROJECT.md §6.1 的 T0）


# --- 功率 --------------------------------------------------------------------


def db_to_linear(value_db):
    """功率比 dB → 线性比值。"""
    return 10.0 ** (np.asarray(value_db, dtype=float) / 10.0)


def linear_to_db(value):
    """线性比值 → dB（功率比口径，即 10·log10）。"""
    return 10.0 * np.log10(np.asarray(value, dtype=float))


def dbm_to_w(value_dbm):
    """dBm → W。"""
    return 1.0e-3 * db_to_linear(value_dbm)


def dbw_to_w(value_dbw):
    """dBW → W。"""
    return db_to_linear(value_dbw)


def w_to_dbm(value_w):
    """W → dBm。"""
    return linear_to_db(np.asarray(value_w, dtype=float) / 1.0e-3)


# --- 长度 --------------------------------------------------------------------


def km_to_m(value_km):
    """km → m。"""
    return np.asarray(value_km, dtype=float) * 1.0e3


def nmi_to_m(value_nmi):
    """海里 → m（1 nmi = 1852 m，精确定义）。"""
    return np.asarray(value_nmi, dtype=float) * 1852.0


# --- 频率 --------------------------------------------------------------------


def mhz_to_hz(value_mhz):
    """MHz → Hz。"""
    return np.asarray(value_mhz, dtype=float) * 1.0e6


def ghz_to_hz(value_ghz):
    """GHz → Hz。"""
    return np.asarray(value_ghz, dtype=float) * 1.0e9


def hz_to_mhz(value_hz):
    """Hz → MHz。"""
    return np.asarray(value_hz, dtype=float) * 1.0e-6


# --- 时间 --------------------------------------------------------------------


def ms_to_s(value_ms):
    """ms → s。"""
    return np.asarray(value_ms, dtype=float) * 1.0e-3


# --- 角度 --------------------------------------------------------------------


def deg_to_rad(value_deg):
    """度 → rad。"""
    return np.deg2rad(np.asarray(value_deg, dtype=float))


def rad_to_deg(value_rad):
    """rad → 度。"""
    return np.rad2deg(np.asarray(value_rad, dtype=float))
