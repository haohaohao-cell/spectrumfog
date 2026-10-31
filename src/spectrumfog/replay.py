"""JSONL 回放：一行一个事件，且**每一行都能独立解析**（PROJECT.md §7.2）。

约定
----
* 第一行是 ``meta``（想定名、seed、dt、pfa、静态几何量），其余行按时间顺序追加。
* 每行都有 ``kind`` 与 ``t``；不允许跨行引用（不写"同上"、不依赖上一行的状态）。
* 写文件用 ``sort_keys=True``，同一份仿真重复跑出来的**字节完全相同**，
  方便直接 diff / 哈希验真。

事件类型
--------
``meta``        仿真级常量（一次）
``state``       某平台在某时刻的位置与辐射源状态（每平台每 tick 一条）
``link``        一把雷达 / 一台干扰机 / 一个目标的几何与物理量（每 tick 一条）
``detection``   雷达对该 tick 的检测裁决（含 Pd 与掷点，保证可复算）
``measurement`` 一条 ESM 观测：AOA / TOA / 频率 / 功率，或一次漏检（T3a）

    注意：PROJECT.md §7.2 的示例把 ESM 观测也叫 ``detection``，但本仓库里
    ``detection`` 已被"雷达对目标的检测"占用（T1 动画依赖该形状），
    而且一个 tick 会有多条 ESM 观测，无法塞进单槽。所以 ESM 观测独立成
    ``measurement``，语义与 §7.2 的 ``detection`` 一致。

``estimate``    融合定位结果：位置、协方差、参与量测数与几何量（T3a）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "EVENT_KINDS",
    "Frame",
    "Replay",
    "write_replay",
    "iter_events",
    "read_replay",
    "build_frames",
]

EVENT_KINDS = ("meta", "state", "link", "detection", "measurement", "estimate", "comms",
                "decoy", "track")


def write_replay(path: str | Path, events: Sequence[Mapping[str, Any]]) -> int:
    """把事件序列写成 JSONL，返回写出的行数。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            if "kind" not in event or "t" not in event:
                raise ValueError(f"事件缺少 kind/t 字段：{event!r}")
            if event["kind"] not in EVENT_KINDS:
                raise ValueError(f"未知事件类型 {event['kind']!r}")
            handle.write(
                json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            handle.write("\n")
    return len(events)


def iter_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读回放，遇到坏行就带上行号报错（不回退、不猜）。"""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} 不是合法 JSON：{exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{source}:{line_number} 不是 JSON 对象")
            if "kind" not in event or "t" not in event:
                raise ValueError(f"{source}:{line_number} 缺少 kind/t 字段")
            if event["kind"] not in EVENT_KINDS:
                raise ValueError(
                    f"{source}:{line_number} 未知事件类型 {event['kind']!r}"
                )
            yield event


def read_replay(path: str | Path) -> list[dict[str, Any]]:
    """整份读进内存（v1 的回放只有几百到几万行，够用）。"""
    return list(iter_events(path))


@dataclass(frozen=True)
class Frame:
    """一个 tick 的完整快照，供 viz / AAR 逐帧使用。"""

    t: float
    positions: dict[str, tuple[float, float]]
    emitters: dict[str, str | None]
    link: dict[str, Any] | None
    detection: dict[str, Any] | None
    measurements: tuple[dict[str, Any], ...] = ()
    estimates: tuple[dict[str, Any], ...] = ()
    comms: tuple[dict[str, Any], ...] = ()
    decoys: tuple[dict[str, Any], ...] = ()
    tracks: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Replay:
    """一份已加载的回放：meta + 按时间排序的帧。"""

    meta: dict[str, Any]
    frames: tuple[Frame, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        events = read_replay(path)
        meta = next((event for event in events if event["kind"] == "meta"), None)
        if meta is None:
            raise ValueError(f"回放缺少 meta 行：{path}")
        payload = [event for event in events if event["kind"] != "meta"]
        return cls(meta=meta, frames=build_frames(payload))

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(frame.t for frame in self.frames)

    @property
    def duration(self) -> float:
        return float(self.frames[-1].t) if self.frames else 0.0


def build_frames(events: Sequence[Mapping[str, Any]]) -> tuple[Frame, ...]:
    """把事件按时间聚合成帧。

    不依赖事件顺序（只依赖 ``t``），这样即使日志被按时间重排也能正确回放。
    """
    buckets: dict[float, dict[str, Any]] = {}
    for event in events:
        t = float(event["t"])
        bucket = buckets.setdefault(
            t,
            {
                "positions": {},
                "emitters": {},
                "link": None,
                "detection": None,
                "measurements": [],
                "estimates": [],
                "comms": [],
                "decoys": [],
                "tracks": [],
            },
        )
        kind = event["kind"]
        if kind == "state":
            pos = event["pos"]
            bucket["positions"][str(event["platform"])] = (
                float(pos[0]),
                float(pos[1]),
            )
            bucket["emitters"][str(event["platform"])] = event.get("emitter")
        elif kind == "link":
            bucket["link"] = dict(event)
        elif kind == "detection":
            bucket["detection"] = dict(event)
        elif kind == "measurement":
            bucket["measurements"].append(dict(event))
        elif kind == "estimate":
            bucket["estimates"].append(dict(event))
        elif kind == "comms":
            bucket["comms"].append(dict(event))
        elif kind == "decoy":
            bucket["decoys"].append(dict(event))
        elif kind == "track":
            bucket["tracks"].append(dict(event))

    return tuple(
        Frame(
            t=t,
            positions=buckets[t]["positions"],
            emitters=buckets[t]["emitters"],
            link=buckets[t]["link"],
            detection=buckets[t]["detection"],
            measurements=tuple(buckets[t]["measurements"]),
            estimates=tuple(buckets[t]["estimates"]),
            comms=tuple(buckets[t]["comms"]),
            decoys=tuple(buckets[t]["decoys"]),
            tracks=tuple(buckets[t]["tracks"]),
        )
        for t in sorted(buckets)
    )
