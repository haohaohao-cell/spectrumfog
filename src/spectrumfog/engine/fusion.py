"""定位融合：测向（AOA）与到达时间差（TDOA）的加权最小二乘。

只做几何，不碰场景模型：输入是若干条量测（传感器位置 + 角度 / 到达时刻），
输出是位置估计、协方差、以及几个解释用的几何量（交会角、GDOP）。

为什么自己做最小二乘而不是调库：这里是二维、两个未知数，Gauss-Newton 十几行就够，
而且能把"协方差怎么来的"写清楚——AAR 要解释"为什么这个时刻被定位"，必须知道
几何从哪里开始变好。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "Fix",
    "Measure",
    "solve_fix",
    "bearing_intersection",
    "DEFAULT_SIGMA_AOA_RAD",
    "DEFAULT_SIGMA_TOA_S",
]

SPEED_OF_LIGHT_MPS = 299_792_458.0

#: 量测精度的兜底值（量测里没带 sigma 时用）
DEFAULT_SIGMA_AOA_RAD = math.radians(2.0)
DEFAULT_SIGMA_TOA_S = 1.0e-7


@dataclass(frozen=True)
class Measure:
    """一条可用于定位的量测。角度用 rad，位置用 m。"""

    sensor_id: str
    sensor_pos: tuple[float, float]
    aoa_rad: float | None = None
    toa_s: float | None = None
    sigma_aoa_rad: float = DEFAULT_SIGMA_AOA_RAD
    sigma_toa_s: float = DEFAULT_SIGMA_TOA_S


@dataclass(frozen=True)
class Fix:
    """一次定位解。"""

    pos: np.ndarray
    covariance: np.ndarray
    n_measurements: int
    sensors: tuple[str, ...]
    geometry_deg: float | None  # 两条测向线的交会角；TDOA 或不足两条时为 None

    @property
    def gdop(self) -> float:
        """几何精度因子：协方差迹的平方根（m）。越小越"硬"。"""
        return float(np.sqrt(np.trace(self.covariance)))


def _wrap(angle: float) -> float:
    """把角度折回 (-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bearing_intersection(
    first_pos: Sequence[float],
    first_aoa: float,
    second_pos: Sequence[float],
    second_aoa: float,
) -> np.ndarray | None:
    """两条测向线的交点；接近平行时返回 None。"""
    p1 = np.asarray(first_pos, dtype=float)
    p2 = np.asarray(second_pos, dtype=float)
    d1 = np.array([math.cos(first_aoa), math.sin(first_aoa)])
    d2 = np.array([math.cos(second_aoa), math.sin(second_aoa)])

    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-6:  # 平行（或反向）
        return None
    delta = p2 - p1
    t1 = (delta[0] * d2[1] - delta[1] * d2[0]) / denominator
    return p1 + t1 * d1


def _initial_guess(measures: Sequence[Measure]) -> np.ndarray:
    aoa = [m for m in measures if m.aoa_rad is not None]
    for i in range(len(aoa)):
        for j in range(i + 1, len(aoa)):
            point = bearing_intersection(
                aoa[i].sensor_pos, aoa[i].aoa_rad, aoa[j].sensor_pos, aoa[j].aoa_rad
            )
            if point is not None:
                return point
    positions = np.array([m.sensor_pos for m in measures], dtype=float)
    return positions.mean(axis=0)


def _residuals(measures: Sequence[Measure], point: np.ndarray) -> np.ndarray:
    """加权残差（除以各自的 sigma，单位变成"多少个 sigma"）。"""
    parts: list[float] = []
    toa_pairs = [m for m in measures if m.toa_s is not None]

    for measure in measures:
        if measure.aoa_rad is None:
            continue
        dx = point[0] - measure.sensor_pos[0]
        dy = point[1] - measure.sensor_pos[1]
        predicted = math.atan2(dy, dx)
        parts.append(_wrap(measure.aoa_rad - predicted) / measure.sigma_aoa_rad)

    if len(toa_pairs) >= 2:
        reference = toa_pairs[0]
        base_range = math.dist(point, reference.sensor_pos)
        for measure in toa_pairs[1:]:
            predicted_delta = (
                math.dist(point, measure.sensor_pos) - base_range
            ) / SPEED_OF_LIGHT_MPS
            observed_delta = measure.toa_s - reference.toa_s
            sigma = math.hypot(measure.sigma_toa_s, reference.sigma_toa_s)
            parts.append((observed_delta - predicted_delta) / sigma)

    return np.asarray(parts, dtype=float)


def solve_fix(
    measures: Sequence[Measure],
    *,
    max_iterations: int = 40,
    tolerance_m: float = 1.0e-3,
) -> Fix | None:
    """加权最小二乘定位。

    残差同时包含测向角与到达时间差；未知量只有 (x, y)（发射时刻在 TDOA 里被差分掉）。
    少于两条可用的测向、且不足两组 TDOA 时，几何上定不了位，返回 None。
    """
    usable = [m for m in measures if m.aoa_rad is not None or m.toa_s is not None]
    n_aoa = sum(1 for m in usable if m.aoa_rad is not None)
    n_toa = sum(1 for m in usable if m.toa_s is not None)
    if n_aoa < 2 and n_toa < 3:
        return None

    point = _initial_guess(usable)
    step = 1.0  # 数值雅可比的差分步长 [m]

    for _ in range(max_iterations):
        base = _residuals(usable, point)
        if base.size == 0:
            return None
        jacobian = np.zeros((base.size, 2))
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = step
            jacobian[:, axis] = (_residuals(usable, point + offset) - base) / step

        # Gauss-Newton：解 (JᵀJ) Δ = -Jᵀr
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ base
        try:
            delta = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta)):
            return None

        point = point + delta
        if float(np.linalg.norm(delta)) < tolerance_m:
            break

    residual = _residuals(usable, point)
    jacobian = np.zeros((residual.size, 2))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        jacobian[:, axis] = (_residuals(usable, point + offset) - residual) / step

    normal = jacobian.T @ jacobian
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None

    # 有冗余时用残差估计方差因子（chi²/dof），否则按单位权处理
    dof = residual.size - 2
    if dof > 0:
        variance_factor = float(residual @ residual) / dof
        covariance = covariance * max(variance_factor, 1e-12)

    if not np.all(np.isfinite(covariance)) or np.trace(covariance) <= 0.0:
        return None

    geometry = None
    bearings = [m for m in usable if m.aoa_rad is not None]
    if len(bearings) == 2:
        separation = abs(_wrap(bearings[0].aoa_rad - bearings[1].aoa_rad))
        geometry = math.degrees(min(separation, math.pi - separation))

    return Fix(
        pos=point,
        covariance=covariance,
        n_measurements=len(usable),
        sensors=tuple(m.sensor_id for m in usable),
        geometry_deg=geometry,
    )
