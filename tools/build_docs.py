"""把 Markdown 编译成静态文档站（不依赖 MkDocs）。

产物写进 ``viz/web/public/docs/``：

    viz/web/public/docs/index.html      概览（README）
    viz/web/public/docs/roadmap.html    路线图（PROJECT.md）
    viz/web/public/docs/physics.html    …其余 docs/*.md
    viz/web/public/docs/images/*        文档配图

Vite 会把 ``public/`` 原样拷进 ``dist/``，所以 GitHub Pages 上就是
``/spectrumfog/docs/``。这样文档站和演示页同源、同一次部署，不需要第二套构建。

用法::

    python tools/build_docs.py            # 生成
    python tools/build_docs.py --check    # 只比对，产物不一致就退出码 1（CI 用）
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "viz" / "web" / "public" / "docs"
ASSETS_DIR = REPO_ROOT / "docs" / "images"


@dataclass(frozen=True)
class Page:
    source: str  # 相对仓库根
    slug: str  # 输出文件名
    nav: str  # 侧栏标签
    tagline: str = ""


PAGES: tuple[Page, ...] = (
    Page("README.md", "index.html", "概览", "从零到跑起来"),
    Page("docs/physics.md", "physics.html", "物理", "公式、假设、缺口"),
    Page("docs/design.md", "design.html", "设计取舍", "为什么这么做"),
    Page("docs/scenario.md", "scenario.html", "想定格式", "YAML schema"),
    Page("docs/metrics.md", "metrics.html", "指标", "Pd / CEP / 暴露"),
    Page("docs/bench.md", "bench.html", "评测", "Gym 环境与 baseline"),
    Page("docs/web.md", "web.html", "网页数据包", "前端不算物理"),
    Page("PROJECT.md", "roadmap.html", "路线图", "里程碑与验收"),
)

#: 不发布到文档站的文件（发布文案属于内部物料）
EXCLUDED = {"docs/launch.md"}

_CSS = """
:root {
  --ink: #2b2f36; --muted: #6b7178; --line: #e3e6ea; --bg: #f7f8fa;
  --accent: #2f6fbf; --code-bg: #f2f4f7;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); line-height: 1.7; }
.layout { display: grid; grid-template-columns: 230px minmax(0, 880px); gap: 36px;
          max-width: 1200px; margin: 0 auto; padding: 28px 24px 64px; }
.side { position: sticky; top: 24px; align-self: start; font-size: 13.5px; }
.side .brand { font-weight: 700; font-size: 15px; margin-bottom: 2px; }
.side .sub { color: var(--muted); font-size: 12px; margin-bottom: 18px; }
.side a { display: block; color: var(--ink); text-decoration: none; padding: 3px 0; }
.side a:hover { color: var(--accent); }
.side a.active { color: var(--accent); font-weight: 600; }
.side .tag { color: var(--muted); font-size: 11.5px; padding-left: 0; }
.side hr { border: 0; border-top: 1px solid var(--line); margin: 14px 0; }
.side .back { color: var(--muted); font-size: 12.5px; }
main { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 28px 34px 40px; }
main > h1:first-child { margin-top: 0; }
h1 { font-size: 26px; } h2 { font-size: 20px; margin-top: 34px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 16.5px; margin-top: 26px; }
a { color: var(--accent); }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 0.92em;
       font-family: "Cascadia Mono", "Consolas", monospace; }
pre { background: #f6f8fa; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
      overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 12.5px; line-height: 1.55; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0; }
th, td { border: 1px solid var(--line); padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
blockquote { margin: 14px 0; padding: 6px 16px; border-left: 3px solid var(--accent);
             color: var(--muted); background: #f8fafc; }
img { max-width: 100%; border-radius: 8px; border: 1px solid var(--line); }
hr { border: 0; border-top: 1px solid var(--line); margin: 26px 0; }
footer { max-width: 1200px; margin: 0 auto; padding: 0 24px 40px; color: var(--muted); font-size: 12px; }
@media (max-width: 820px) {
  .layout { grid-template-columns: 1fr; }
  .side { position: static; }
  main { padding: 20px; }
}
"""

_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · SpectrumFog</title>
<style>{css}</style>
</head>
<body>
<div class="layout">
  <aside class="side">
    <div class="brand">SpectrumFog</div>
    <div class="sub">频谱迷雾 · 文档</div>
    {nav}
    <hr>
    <a class="back" href="../index.html">← 打开交互演示</a>
    <a class="back" href="https://github.com/haohaohao-cell/spectrumfog">GitHub 仓库</a>
  </aside>
  <main>
{content}
  </main>
</div>
<footer>参数全部虚构 · 仅用于教学 / 研究仿真 · Apache-2.0 + CC BY 4.0</footer>
</body>
</html>
"""


def _title_of(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _slug_map() -> dict[str, str]:
    return {page.source: page.slug for page in PAGES}


def _rewrite_links(rendered: str, source: str, slugs: dict[str, str]) -> str:
    """把 .md 链接改成本站的 .html，并把图片路径收敛到 images/。"""
    source_dir = (REPO_ROOT / source).parent

    def resolve(target: str) -> str:
        if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return target
        path, _, anchor = target.partition("#")
        if not path:
            return target
        candidate = (source_dir / path).resolve()
        try:
            relative = candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return target
        if candidate.suffix in (".png", ".gif", ".jpg", ".jpeg", ".svg"):
            if relative.startswith("docs/images/"):
                return "images/" + candidate.name + (f"#{anchor}" if anchor else "")
            return target
        if relative in slugs:
            return slugs[relative] + (f"#{anchor}" if anchor else "")
        return target

    def replace(match: re.Match[str]) -> str:
        attr, value = match.group(1), match.group(2)
        return f'{attr}="{resolve(value)}"'

    return re.sub(r'(href|src)="([^"]+)"', replace, rendered)


def render_page(page: Page, slugs: dict[str, str]) -> str:
    source_path = REPO_ROOT / page.source
    text = source_path.read_text(encoding="utf-8")
    title = _title_of(text, page.nav)
    rendered = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
        output_format="html5",
    )
    rendered = _rewrite_links(rendered, page.source, slugs)

    links = []
    for entry in PAGES:
        active = " active" if entry.slug == page.slug else ""
        links.append(
            f'<a class="nav{active}" href="{entry.slug}">{html.escape(entry.nav)}</a>'
            f'<span class="tag">{html.escape(entry.tagline)}</span>'
        )
    return _TEMPLATE.format(
        title=html.escape(title),
        css=_CSS,
        nav="\n    ".join(links),
        content=rendered,
    )


def build(target: Path) -> list[Path]:
    from markdown.extensions.toc import TocExtension  # noqa: F401  (确保扩展可用)

    slugs = _slug_map()
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for page in PAGES:
        destination = target / page.slug
        destination.write_text(render_page(page, slugs), encoding="utf-8")
        written.append(destination)

    images_target = target / "images"
    if images_target.exists():
        shutil.rmtree(images_target)
    if ASSETS_DIR.is_dir():
        shutil.copytree(ASSETS_DIR, images_target)
    return written


def _same_tree(left: Path, right: Path) -> bool:
    left_files = sorted(p.relative_to(left) for p in left.rglob("*") if p.is_file())
    right_files = sorted(p.relative_to(right) for p in right.rglob("*") if p.is_file())
    if left_files != right_files:
        return False
    return all((left / rel).read_bytes() == (right / rel).read_bytes() for rel in left_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 SpectrumFog 文档站")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已提交的产物是否与源码一致（CI 用）",
    )
    args = parser.parse_args(argv)

    if not args.check:
        written = build(OUTPUT_DIR)
        total = sum(page.stat().st_size for page in OUTPUT_DIR.rglob("*") if page.is_file())
        print(f"文档站 → {OUTPUT_DIR}  （{len(written)} 页，{total / 1024:.0f} KB 含配图）")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "docs"
        build(fresh)
        if not OUTPUT_DIR.exists():
            print("[错误] 还没有产物，先跑 python tools/build_docs.py", flush=True)
            return 1
        if not _same_tree(fresh, OUTPUT_DIR):
            print(
                "[错误] 文档站产物与 Markdown 源码不一致，请跑 python tools/build_docs.py 后提交",
                flush=True,
            )
            return 1
    print("文档站产物与源码一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
