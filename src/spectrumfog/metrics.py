"""指标口径（PROJECT.md §8 / docs/metrics.md）。

只做纯计算：输入的误差、CEP 序列、时间步长都是已经算好的数字。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "cep_from_samples",
    "cep_from_covariance",
    "ExposureWindow",
    "exposure_windows",
    "t_expose",
]


def cep_from_samples(errors_m: np.ndarray, quantile: float = 0.5) -> float:
    """样本分位数的 CEP。

    docs/metrics.md §2 规定：指标本身用**样本分位数**，不用 1.177σ 之类的近似。
    """
    values = np.asarray(errors_m, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("没有有效的误差样本")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile 必须落在 (0,1)，收到 {quantile!r}")
    return float(np.quantile(values, quantile))


def cep_from_covariance(
    covariance: np.ndarray, *, samples: int = 8000, seed: int = 0
) -> tuple[float, float]:
    """由 2×2 位置协方差估计 (CEP50, CEP90)。

    指标本身用样本误差（``cep_from_samples``）；这里是从**估计器自报的协方差**
    反推一个 CEP，用来和真实误差对比——两者差太多就说明滤波器过于自信或过于保守。

    用固定 seed 的蒙特卡洛，结果可复现；8000 个样本时 50% 分位的标准误约 0.6%。
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"协方差必须是 2×2，收到 {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("协方差里有非有限值")

    # 特征分解后再采样：避免要求输入严格正定
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    rng = np.random.default_rng(seed)
    offsets = rng.standard_normal((samples, 2)) @ root.T
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    return (
        float(np.quantile(radii, 0.5)),
        float(np.quantile(radii, 0.9)),
    )


@dataclass(frozen=True)
class ExposureWindow:
    """一段"被有效定位"的时间窗。"""

    start_s: float
    end_s: float
    peak_cep_m: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def exposure_windows(
    times_s: np.ndarray,
    cep_m: np.ndarray,
    threshold_m: float,
    *,
    dt: float = 1.0,
) -> tuple[ExposureWindow, ...]:
    """连续满足 CEP ≤ 门槛的区间。

    每个 tick 代表区间 ``[t, t + dt)``（docs/metrics.md §3 的口径），
    否则"只暴露了一帧"会被算成 0 秒。
    """
    times = np.asarray(times_s, dtype=float)
    ceps = np.asarray(cep_m, dtype=float)
    if times.shape != ceps.shape:
        raise ValueError("times 与 cep 长度不一致")
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，收到 {dt!r}")

    windows: list[ExposureWindow] = []
    start: float | None = None
    peak = 0.0
    for index in range(times.size):
        exposed = bool(np.isfinite(ceps[index])) and ceps[index] <= threshold_m
        if exposed:
            if start is None:
                start = float(times[index])
                peak = float(ceps[index])
            else:
                peak = max(peak, float(ceps[index]))
        elif start is not None:
            windows.append(
                ExposureWindow(
                    start_s=start, end_s=float(times[index - 1]) + dt, peak_cep_m=peak
                )
            )
            start = None
    if start is not None:
        windows.append(
            ExposureWindow(start_s=start, end_s=float(times[-1]) + dt, peak_cep_m=peak)
        )
    return tuple(windows)


def t_expose(
    times_s: np.ndarray,
    cep_m: np.ndarray,
    threshold_m: float,
    *,
    dt: float = 1.0,
) -> float:
    """暴露时长 T_expose [s]：CEP ≤ 门槛的那些 tick 的总时长（docs/metrics.md §3）。"""
    return sum(
        window.duration_s
        for window in exposure_windows(times_s, cep_m, threshold_m, dt=dt)
    )
