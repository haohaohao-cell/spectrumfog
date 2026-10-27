"""可视化（表现层）。

* ``static/``：matplotlib 出图（T0 / T1），只读物理与 replay，不回写内核。
* ``web/``：Vite + React + deck.gl（T2 才建，现在不存在）。

本包与 ``spectrumfog``（内核）的关系是单向的：viz 可以 import 内核，
内核绝不 import viz（PROJECT.md §2 决策 2）。
"""
