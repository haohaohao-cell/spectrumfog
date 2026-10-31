"""生成「处理增益」对比图（docs/images/processing_gain.png）。

数据来自 07_gain 与 01_duel 的实测（601 帧），不是手画：
处理增益 0 / 20 / 40 dB 时，雷达在 601 帧里命中目标的次数分别是 0 / 2 / 505。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体：Windows 上优先微软雅黑，Linux/CI 上退回 Noto/思源，最后才用 DejaVu
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "processing_gain.png"

GAINS = ("0 dB\n(01_duel)", "20 dB", "40 dB\n(07_gain)")
HITS = (0, 2, 505)
COLORS = ("#b23c3c", "#d08a2c", "#2f6fbf")


def main() -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)

    bars = ax.bar(GAINS, HITS, color=COLORS, width=0.55)
    ax.set_ylim(0, 601)
    ax.set_ylabel("601 帧里雷达命中目标的次数", fontsize=11)
    ax.set_xlabel("雷达的处理增益（脉冲压缩 + 相干积累）", fontsize=11)
    ax.set_title(
        "同样是 90 W 干扰机在 30 km：20 dB 处理增益几乎救不了雷达",
        fontsize=12,
        pad=14,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for bar, hits in zip(bars, HITS, strict=True):
        label = f"{hits}" if hits else "0"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.annotate(
        "功率 ×100、烧穿距离 ×10\n但 601 帧只救回 2 帧",
        xy=(1, 2),
        xytext=(0.45, 330),
        fontsize=9.5,
        color="#6b5a2e",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#6b5a2e", "lw": 0.9},
    )

    fig.text(
        0.99,
        0.01,
        "fictional parameters · educational simulation · SpectrumFog",
        ha="right",
        fontsize=7,
        color="#9aa0a8",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(main())
