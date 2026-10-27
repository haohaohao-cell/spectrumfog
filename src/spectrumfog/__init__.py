"""SpectrumFog / 频谱迷雾 —— 部分可观测的电磁对抗仿真内核。

设计约束（见 PROJECT.md §2）：
  * 确定性优先：固定步长 + seeded RNG + 事件日志。
  * 引擎与表现分离：本包不 import 任何 UI / 绘图库。
  * 内部一律 SI 单位（W / m / Hz / s / rad），dB 只活在边界上。

模块地图（T0 只落地 physics，其余按里程碑逐步补齐）：
  units       SI 常量 + 单位换算
  physics     雷达方程、J/S、烧穿距离、检测概率
"""

__version__ = "0.2.0"
