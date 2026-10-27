"""频段重叠判据。

v1 假设矩形谱：两个发射只要谱有交叠就算"同频"，可以互相干扰。
多频段想定（03_triad）靠这一条区分"谁打谁"；

不是物理上的精确频谱——不做滤波器响应、邻道抑制、互调。
"""

from __future__ import annotations

__all__ = ["in_band"]


def in_band(freq_a: float, bw_a: float, freq_b: float, bw_b: float) -> bool:
    """两个中心频率 / 带宽是否重叠（矩形谱）。"""
    return abs(float(freq_a) - float(freq_b)) <= 0.5 * (float(bw_a) + float(bw_b))
