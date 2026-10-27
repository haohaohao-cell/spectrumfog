"""物理内核：只做纯数学，不持有状态、不碰 IO、不 import UI 库。

T0 已落地：
  radar         雷达方程、干信比 J/S、烧穿距离 R_bt、Swerling I 检测概率、检测距离
  propagation   自由空间路径损耗（Friis）
  comms         链路预算、Eb/N0、抗干扰余量

T1+ 待补：
  propagation   可选的绕射 / 大气吸收项（当前只有自由空间）
"""

from __future__ import annotations

__all__ = ["radar", "propagation", "comms"]
