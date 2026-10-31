"""想定集合（suite）、提交物与排行榜。

排行榜要做到"可比"，必须锁三样东西：**哪些想定、哪些种子、哪些策略**。
``scenarios/suite_v1.yaml`` 就是这把尺子；提交物里带上每份想定文件的 sha256，
所以别人改一个数字、换一个版本，校验就能立刻发现——不需要服务器。

没有后端的分工：

    spectrumfog bench --suite ... --submit   →  提交物 JSON（本地跑）
    spectrumfog bench --verify   FILE        →  校验提交物与当前想定是否一致
    spectrumfog bench --leaderboard DIR      →  从 submissions/*.json 汇总排名
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from ..bench.runner import BenchRow

__all__ = [
    "Suite",
    "load_suite",
    "build_submission",
    "verify_submission",
    "render_leaderboard",
    "write_submission",
]

SUBMISSION_SCHEMA = "spectrumfog.submission/2"


@dataclass(frozen=True)
class Suite:
    """一套固定的评测条件。"""

    name: str
    version: int
    seeds: tuple[int, ...]
    policies: tuple[str, ...]
    scenario_paths: tuple[str, ...]
    created: str = ""
    description: str = ""

    @property
    def episodes(self) -> int:
        return len(self.seeds)

    def scenario_hashes(self) -> dict[str, str]:
        """每份想定的**内容**指纹（规范化 JSON 的 sha256）。

        刻意不哈希文件字节：改注释、调缩进、换行符都不该让别人的提交物失效，
        真正要防的是"改了一个参数"。想定解析成模型之后的规范 JSON 才是物理输入。
        """
        from ..model.scenario import load_scenario

        hashes: dict[str, str] = {}
        for relative in self.scenario_paths:
            path = Path(relative)
            if not path.is_file():
                raise FileNotFoundError(f"想定文件不存在：{relative}")
            scenario = load_scenario(path)
            canonical = json.dumps(
                scenario.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
            )
            hashes[relative] = (
                "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            )
        return hashes


def load_suite(path: str | Path) -> Suite:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"suite 文件不存在：{source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source} 顶层必须是映射")

    try:
        scenarios = raw["scenarios"]
        seeds = raw["seeds"]
        policies = raw["policies"]
    except KeyError as exc:
        raise ValueError(f"{source} 缺少字段：{exc}") from exc

    if not scenarios:
        raise ValueError(f"{source} 的 scenarios 不能为空")
    if not seeds:
        raise ValueError(f"{source} 的 seeds 不能为空")
    if not policies:
        raise ValueError(f"{source} 的 policies 不能为空")

    return Suite(
        name=str(raw.get("name", source.stem)),
        version=int(raw.get("version", 1)),
        seeds=tuple(int(seed) for seed in seeds),
        policies=tuple(str(policy) for policy in policies),
        scenario_paths=tuple(str(item) for item in scenarios),
        created=str(raw.get("created", "")),
        description=str(raw.get("description", "")).strip(),
    )


def _environment() -> dict[str, str]:
    info = {"python": platform.python_version()}
    for module_name in ("numpy", "gymnasium"):
        try:
            module = __import__(module_name)
            info[module_name] = str(getattr(module, "__version__", "?"))
        except ImportError:  # pragma: no cover - 取决于环境
            info[module_name] = "missing"
    return info


def build_submission(
    suite: Suite,
    rows: Sequence[BenchRow],
    *,
    label: str,
    created: str | None = None,
) -> dict[str, Any]:
    """把一次评测打包成可校验、可比较的提交物。"""
    if not rows:
        raise ValueError("没有评测结果，无法生成提交物")

    score = sum(row.reward_mean for row in rows) / len(rows)
    return {
        "schema": SUBMISSION_SCHEMA,
        "label": label,
        "created": created
        or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "seeds": list(suite.seeds),
            "policies": list(suite.policies),
        },
        "scenarios": suite.scenario_hashes(),
        "score": round(score, 4),
        "rows": [asdict(row) for row in rows],
        "environment": _environment(),
    }


def verify_submission(payload: dict[str, Any], suite: Suite) -> list[str]:
    """校验提交物与当前想定集合是否一致，返回问题列表（空 = 通过）。"""
    problems: list[str] = []

    if payload.get("schema") != SUBMISSION_SCHEMA:
        problems.append(f"schema 不是 {SUBMISSION_SCHEMA}：{payload.get('schema')!r}")

    suite_block = payload.get("suite") or {}
    if suite_block.get("name") != suite.name:
        problems.append(f"suite 名不符：{suite_block.get('name')!r} != {suite.name!r}")
    if int(suite_block.get("version", -1)) != suite.version:
        problems.append(
            f"suite 版本不符：{suite_block.get('version')!r} != {suite.version}"
        )
    if list(suite_block.get("seeds", [])) != list(suite.seeds):
        problems.append("种子集合与当前 suite 不一致")
    if list(suite_block.get("policies", [])) != list(suite.policies):
        problems.append("策略集合与当前 suite 不一致")

    current = suite.scenario_hashes()
    submitted = payload.get("scenarios") or {}
    if set(submitted) != set(current):
        problems.append("想定文件清单与当前 suite 不一致")
    else:
        for relative, digest in current.items():
            if submitted[relative] != digest:
                problems.append(f"想定内容已改动：{relative}")

    rows = payload.get("rows") or []
    expected = len(suite.scenario_paths) * len(suite.policies)
    if len(rows) != expected:
        problems.append(f"结果行数不符：{len(rows)} != {expected}")
    return problems


def write_submission(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_submissions(directory: str | Path) -> list[dict[str, Any]]:
    folder = Path(directory)
    if not folder.is_dir():
        return []
    submissions = []
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("schema") == SUBMISSION_SCHEMA:
            submissions.append(payload)
    return submissions


def render_leaderboard(
    submissions: Iterable[dict[str, Any]], *, title: str = "排行榜"
) -> str:
    """把提交物渲染成 markdown 表（按得分降序）。"""
    entries = sorted(
        submissions, key=lambda item: float(item.get("score", -1e9)), reverse=True
    )
    lines = [
        f"# {title}",
        "",
        "同一把尺子：想定集合、种子、策略都固定（见 `scenarios/suite_v1.yaml`）。",
        "得分 = 各（想定 × 策略）平均奖励；奖励 = 干扰/压制有效 +1、被有效定位 −1。",
        "",
        "| # | 提交 | 得分 | 想定数 | 环境 | 日期 |",
        "|---|---|---|---|---|---|",
    ]
    if not entries:
        lines.append("| — | *还没有提交* | — | — | — | — |")
    for index, entry in enumerate(entries, start=1):
        suite = entry.get("suite") or {}
        rows = entry.get("rows") or []
        scenario_count = len({row.get("scenario") for row in rows})
        environment = entry.get("environment") or {}
        lines.append(
            f"| {index} | `{entry.get('label', '?')}` | "
            f"{float(entry.get('score', 0.0)):.2f} | {scenario_count} | "
            f"py{environment.get('python', '?')} / gym{environment.get('gymnasium', '?')} | "
            f"{str(entry.get('created', ''))[:10]} |"
        )
    lines += [
        "",
        "## 怎么提交",
        "",
        "```bash",
        "spectrumfog bench --suite scenarios/suite_v1.yaml --submit --label 你的名字",
        "spectrumfog bench --verify submissions/你的名字-*.json   # 自检",
        "```",
        "",
        "把生成的 JSON 放进 `submissions/` 提 PR。没有服务器：",
        "提交物自带每份想定文件的 sha256，`--verify` 一跑就知道结果是不是在**当前**想定上跑出来的。",
    ]
    return "\n".join(lines) + "\n"
