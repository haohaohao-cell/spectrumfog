"""命令行入口（T0）：一条命令跑完物理闭环并出三张图。

用法::

    spectrumfog run scenarios/01_duel.yaml
    python -m spectrumfog run scenarios/01_duel.yaml --out out --pfa 1e-6

CLI 是"表现层"：只读想定、调用 ``physics``、打印、把出图交给 ``viz``。
它不实现任何物理公式——公式只有 ``spectrumfog.physics`` 一份（§2 决策 3）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import __version__, aar
from .engine.sim import Simulation
from .model.scenario import Scenario, load_scenario
from .physics import radar
from .replay import Replay, write_replay
from .units import linear_to_db
from .webexport import build_web_bundle, write_web_bundle

_FOOTNOTE = "参数为虚构值，仅用于教学 / 研究仿真（PROJECT.md §12）"


# --- 格式化小工具（只在这里把 SI 换成习惯单位，内核里永远是 SI） ---------------


def _fmt_power(watts: float) -> str:
    if watts >= 1.0e3:
        return f"{watts / 1.0e3:.1f} kW"
    if watts >= 1.0:
        return f"{watts:.1f} W"
    if watts >= 1.0e-3:
        return f"{watts * 1.0e3:.2f} mW"
    return f"{watts:.3g} W"


def _fmt_freq(hz: float) -> str:
    if hz >= 1.0e9:
        return f"{hz / 1.0e9:.2f} GHz"
    if hz >= 1.0e6:
        return f"{hz / 1.0e6:.2f} MHz"
    if hz >= 1.0e3:
        return f"{hz / 1.0e3:.2f} kHz"
    return f"{hz:.0f} Hz"


def _fmt_km(m: float) -> str:
    return f"{m / 1.0e3:.2f} km" if m >= 1.0e3 else f"{m:.1f} m"


def _fmt_gain_db(linear: float) -> str:
    return f"{10.0 * math.log10(linear):.1f} dBi"


def _fmt_dbm(watts: float) -> str:
    return f"{float(linear_to_db(watts * 1.0e3)):.1f} dBm"


# --- 子命令 ------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    try:
        radar_platform, radar_emitter = scenario.primary_radar(args.radar)
        jammer_platform, jammer_emitter = scenario.primary_jammer(args.jammer)
        params = scenario.link_params(args.radar, args.jammer)
        standoff_m = scenario.standoff_m(args.radar, args.jammer)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    pfa = args.pfa
    pd_threshold = args.pd_threshold
    environment = scenario.environment

    r_bt = float(radar.burn_through_range(**_burn_through_kwargs(params)))
    pn_w = float(
        radar.noise_power(
            bw_hz=params["bw_r_hz"],
            noise_figure=environment.noise_figure,
            temperature_k=environment.temperature_k,
        )
    )
    detection_kwargs = {
        "pt_w": params["pt_w"],
        "gain": params["gain"],
        "sigma_m2": params["sigma_m2"],
        "lam": params["lam"],
        "bw_r_hz": params["bw_r_hz"],
        "noise_figure": environment.noise_figure,
        "temperature_k": environment.temperature_k,
        "loss": params["loss"],
    }
    r_det_clean = float(radar.detection_range(pd_threshold, pfa, **detection_kwargs))
    try:
        r_det_jammed = float(
            radar.detection_range(
                pd_threshold,
                pfa,
                pj_w=params["pj_w"],
                gj=params["gj"],
                bw_j_hz=params["bw_j_hz"],
                **detection_kwargs,
            )
        )
    except ValueError:
        r_det_jammed = None  # 全距离都被压到 Pd 门槛以下

    pr_w = float(
        radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=standoff_m,
            loss=params["loss"],
        )
    )
    pj_rx_w = float(
        radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=standoff_m,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
    )
    js = float(radar.js_ratio(pr_w=pr_w, pj_rx_w=pj_rx_w))
    pd_at_standoff = float(radar.pd_from_powers(pfa, pr_w, pn_w, pj_rx_w))

    # --- 报告 -------------------------------------------------------------
    meta = scenario.meta
    print("SpectrumFog · T0 物理闭环（内部 SI，输出换算成习惯单位）")
    print(
        f"想定 {meta.name}   种子 {meta.seed}   "
        f"步长 {meta.dt:g} s   时长 {meta.duration:g} s"
    )
    print()
    print(
        f"  雷达   {radar_platform.id:<10s} {_fmt_freq(radar_emitter.freq_hz):>9s}   "
        f"Pt {_fmt_power(radar_emitter.pt_w):>10s}   "
        f"G {_fmt_gain_db(radar_emitter.gain):>9s}   "
        f"Br {_fmt_freq(radar_emitter.bw_hz):>9s}   "
        f"σ {radar_emitter.sigma_m2:.2f} m²   "
        f"L {10.0 * math.log10(radar_emitter.loss):.1f} dB"
    )
    print(
        f"  干扰   {jammer_platform.id:<10s} {_fmt_freq(jammer_emitter.freq_hz):>9s}   "
        f"Pj {_fmt_power(jammer_emitter.pj_w):>10s}   "
        f"Gj {_fmt_gain_db(jammer_emitter.gain):>9s}   "
        f"Bj {_fmt_freq(jammer_emitter.bw_hz):>9s}   "
        f"站距 {_fmt_km(standoff_m)}"
    )
    print()
    print(f"  噪声底 Pn          {pn_w:.4e} W  ({_fmt_dbm(pn_w)})")
    print(f"  烧穿距离 R_bt      {_fmt_km(r_bt)}   (J/S = 1)")
    print(
        f"  检测距离 R_det     {_fmt_km(r_det_clean)}   "
        f"(Pd = {pd_threshold:.2f}, Pfa = {pfa:g}, 仅热噪声)"
    )
    if r_det_jammed is None:
        print(f"  干扰下检测距离     无（全距离 Pd < {pd_threshold:.2f}）")
    else:
        print(
            f"  干扰下检测距离     {_fmt_km(r_det_jammed)}   "
            f"(Pd = {pd_threshold:.2f}, 含干扰)"
        )
    verdict = "干扰压制（J/S > 1）" if js >= 1.0 else "雷达可检测（J/S < 1）"
    print(
        f"  {_fmt_km(standoff_m)} 处"
        f"        J/S = {js:.3g} ({10.0 * math.log10(js):.1f} dB)   "
        f"Pd = {pd_at_standoff:.2e}   → {verdict}"
    )
    print()

    if args.no_figures:
        print(_FOOTNOTE)
        return 0

    plot_all = _load_plotter()
    if plot_all is None:
        print(
            "[警告] 找不到 viz.static.plot（matplotlib 未安装或不在 sys.path）。跳过出图。",
            file=sys.stderr,
        )
        return 0

    paths = plot_all(
        params,
        out_dir=args.out,
        stem=meta.name,
        pfa=pfa,
        pd_threshold=pd_threshold,
        noise_figure=environment.noise_figure,
        temperature_k=environment.temperature_k,
    )
    print("  图：")
    for path in paths:
        print(f"    {path}")
    print()
    print(_FOOTNOTE)
    return 0


def _burn_through_kwargs(params: dict[str, float]) -> dict[str, float]:
    """``burn_through_range`` 不接受 ``lam``（自屏蔽下波长相消）。"""
    return {key: value for key, value in params.items() if key != "lam"}


def _load_plotter():
    """延迟导入表现层：内核与 CLI 在没装 matplotlib 时也能跑。"""
    try:
        from viz.static.plot import plot_all
    except ImportError:
        return None
    return plot_all


def _load_animator():
    """延迟导入动画（viz 只读 replay，不回写）。"""
    try:
        from viz.static.animate import animate_file
    except ImportError:
        return None
    return animate_file


# --- sim：确定性动态仿真 ------------------------------------------------------


def _sim(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    try:
        result = Simulation(
            scenario, pfa=args.pfa, pd_threshold=args.pd_threshold
        ).run()
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    replay_path = (
        args.replay
        if args.replay is not None
        else args.out / f"{scenario.meta.name}.jsonl"
    )
    try:
        n_lines = write_replay(replay_path, result.events)
    except (OSError, ValueError) as exc:
        print(f"[错误] 写回放失败：{exc}", file=sys.stderr)
        return 1

    summary = result.summary
    meta = result.meta
    print("SpectrumFog · T1 动态仿真（固定步长 + seeded RNG + 事件日志）")
    print(
        f"想定 {meta['scenario']}   种子 {meta['seed']}   "
        f"dt {meta['dt']:g} s   时长 {meta['duration']:g} s   "
        f"{summary['n_ticks']} 帧"
    )
    print()
    print(
        f"  链路   {meta['radar']} → {meta['target']}   干扰 {meta['jammer']}   "
        f"Pfa {meta['pfa']:g}   门槛 Pd {meta['pd_threshold']:.2f}"
    )
    print(
        f"  无干扰探测距离     {_fmt_km(summary['unjammed_detection_range_m'])}   "
        f"(Pd = {meta['pd_threshold']:.2f})"
    )
    print(
        f"  干扰机距离         {_fmt_km(summary['jammer_range_start_m'])} → "
        f"{_fmt_km(summary['jammer_range_end_m'])}"
    )
    print(
        f"  目标处 Pd          {summary['pd_max']:.3f} → {summary['pd_min']:.2e}"
    )
    flip = summary["pd_below_threshold_at_s"]
    print(
        f"  Pd 跌破门槛        "
        + ("未跌破（全程可检测）" if flip is None else f"t = {flip:.0f} s")
    )
    print(
        f"  单次扫描命中率     {summary['scan_detection_fraction'] * 100:.1f}%"
        "（抽样结果，不等于是否丢失）"
    )
    print()
    print(f"  replay → {replay_path}   （{n_lines} 行，一行一事件）")

    if args.gif:
        animator = _load_animator()
        if animator is None:
            print(
                "[警告] 找不到 viz.static.animate（matplotlib 未安装？）。跳过动画。",
                file=sys.stderr,
            )
            return 0
        gif_path = args.out / f"{scenario.meta.name}.gif"
        animator(replay_path, gif_path, fps=args.fps)
        print(f"  gif    → {gif_path}")

    print()
    print(_FOOTNOTE)
    return 0


# --- animate：读 replay 出 GIF ------------------------------------------------


def _animate(args: argparse.Namespace) -> int:
    try:
        replay = Replay.load(args.replay)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    animator = _load_animator()
    if animator is None:
        print(
            "[警告] 找不到 viz.static.animate（matplotlib 未安装？）。跳过动画。",
            file=sys.stderr,
        )
        return 0

    out_path = args.out or args.replay.with_suffix(".gif")
    animator(args.replay, out_path, fps=args.fps, stride=args.stride)
    print(
        f"{replay.meta.get('scenario', 'replay')}："
        f"{len(replay.frames)} 帧 → {out_path}（{args.fps} fps）"
    )
    return 0


# --- export-web：给 T2 前端编译数据包 ----------------------------------------


def _export_web(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    try:
        bundle = build_web_bundle(
            scenario,
            pfa=args.pfa,
            pd_threshold=args.pd_threshold,
            samples=args.samples,
        )
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    out_path = args.out or args.out_dir / f"{scenario.meta.name}.web.json"
    try:
        write_web_bundle(out_path, bundle)
    except OSError as exc:
        print(f"[错误] 写数据包失败：{exc}", file=sys.stderr)
        return 1

    grid = bundle["grid"]
    size_kb = out_path.stat().st_size / 1024.0
    print("SpectrumFog · Web 数据包（物理只在这里算一次，前端只插值）")
    print(
        f"  想定 {bundle['scenario']}   Pfa {bundle['pfa']:g}   "
        f"门槛 Pd {bundle['pd_threshold']:.2f}"
    )
    print(
        f"  二维表 {grid['rt_m']['n']}×{grid['rj_m']['n']}："
        f"Rt {grid['rt_m']['min']:.0f}–{grid['rt_m']['max']:.0f} m，"
        f"Rj {grid['rj_m']['min']:.0f}–{grid['rj_m']['max']:.0f} m（对数网格）"
    )
    print(
        f"  曲线   {len(bundle['rings']['jammer_range_m'])} 点："
        "检测边界 / 烧穿环 / 目标处 Pd 与 J-S"
    )
    print(f"  → {out_path}  （{size_kb:.0f} KB）")
    return 0


# --- aar：复盘报告 -----------------------------------------------------------


def _aar(args: argparse.Namespace) -> int:
    try:
        replay = Replay.load(args.replay)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    try:
        report = aar.analyse(
            replay, threshold_m=args.cep_threshold, emitter=args.target
        )
    except ValueError as exc:
        print(f"[错误] 复盘失败：{exc}", file=sys.stderr)
        return 1

    out_path = args.out or args.replay.with_name(f"{args.replay.stem}_aar.html")
    try:
        aar.write_html(out_path, aar.render_html(report, replay))
    except OSError as exc:
        print(f"[错误] 写报告失败：{exc}", file=sys.stderr)
        return 1

    print("SpectrumFog · AAR 复盘（只读 replay，不重算物理）")
    print(f"  想定 {replay.meta.get('scenario', '?')}   目标 {report.emitter}   "
          f"{report.n_ticks} 帧   解出位置 {report.n_fixes} 帧")
    print()
    print(f"  {report.sentence}")
    print()
    print(
        f"  首次有效定位   "
        + ("无（未跌破门槛）" if report.first_effective_s is None
           else f"{report.first_effective_s:.0f} s，CEP {report.first_effective_cep_m:.0f} m")
    )
    print(
        f"  最精确的一次   "
        + ("—" if report.best_cep_m is None
           else f"{report.best_cep_m:.0f} m（t = {report.best_cep_s:.0f} s）")
    )
    print(f"  暴露时长       {report.t_expose_s:.0f} s（门槛 {report.threshold_m:.0f} m）")
    if report.timeline:
        print("  关键事件：")
        for event in report.timeline[:6]:
            print(f"    t = {event.t_s:5.0f} s   {event.kind}   {event.detail}")
    print()
    print(f"  报告 → {out_path}")
    print()
    print(_FOOTNOTE)
    return 0


# --- 参数解析 ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectrumfog",
        description="频谱迷雾 —— 部分可观测电磁对抗仿真（T0：物理闭环）",
    )
    parser.add_argument(
        "--version", action="version", version=f"spectrumfog {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="加载想定、算物理指标、出三张图"
    )
    run.add_argument("scenario", type=Path, help="想定 YAML 路径")
    run.add_argument(
        "--out", type=Path, default=Path("out"), help="出图目录（默认 out/）"
    )
    run.add_argument(
        "--pfa", type=float, default=1.0e-6, help="虚警概率（默认 1e-6）"
    )
    run.add_argument(
        "--pd-threshold",
        type=float,
        default=0.5,
        help="算检测距离用的 Pd 门槛（默认 0.5）",
    )
    run.add_argument("--radar", default=None, metavar="PLATFORM_ID", help="指定雷达平台")
    run.add_argument("--jammer", default=None, metavar="PLATFORM_ID", help="指定干扰平台")
    run.add_argument(
        "--no-figures", action="store_true", help="只打印指标，不出图"
    )

    sim = subparsers.add_parser(
        "sim", help="跑确定性动态仿真，写 replay JSONL（可选出 GIF）"
    )
    sim.add_argument("scenario", type=Path, help="想定 YAML 路径")
    sim.add_argument(
        "--out", type=Path, default=Path("out"), help="输出目录（默认 out/）"
    )
    sim.add_argument(
        "--replay", type=Path, default=None, help="replay 文件路径（默认 out/<name>.jsonl）"
    )
    sim.add_argument("--pfa", type=float, default=1.0e-6, help="虚警概率（默认 1e-6）")
    sim.add_argument(
        "--pd-threshold", type=float, default=0.5, help="检测门槛（默认 0.5）"
    )
    sim.add_argument("--gif", action="store_true", help="仿真后直接出 GIF")
    sim.add_argument("--fps", type=int, default=12, help="GIF 帧率（默认 12）")

    animate = subparsers.add_parser("animate", help="读 replay JSONL，出俯视动画 GIF")
    animate.add_argument("replay", type=Path, help="replay 文件路径")
    animate.add_argument(
        "-o", "--out", type=Path, default=None, help="GIF 输出路径（默认与 replay 同名）"
    )
    animate.add_argument("--fps", type=int, default=12, help="帧率（默认 12）")
    animate.add_argument(
        "--stride", type=int, default=1, help="每隔多少帧画一帧（默认 1，用于压体积）"
    )

    export_web = subparsers.add_parser(
        "export-web", help="把想定编译成前端数据包（T2 用，无后端）"
    )
    export_web.add_argument("scenario", type=Path, help="想定 YAML 路径")
    export_web.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="输出 JSON 路径（默认 out/<name>.web.json）",
    )
    export_web.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="默认输出目录"
    )
    export_web.add_argument("--pfa", type=float, default=1.0e-6, help="虚警概率")
    export_web.add_argument(
        "--pd-threshold", type=float, default=0.5, help="检测门槛（默认 0.5）"
    )
    export_web.add_argument(
        "--samples", type=int, default=97, help="二维表每轴的采样数（默认 97）"
    )

    aar_parser = subparsers.add_parser(
        "aar", help="复盘一份 replay，生成时间线 + 归因 + 单页 HTML 报告"
    )
    aar_parser.add_argument("replay", type=Path, help="replay JSONL 路径")
    aar_parser.add_argument(
        "-o", "--out", type=Path, default=None, help="HTML 输出路径（默认与 replay 同名）"
    )
    aar_parser.add_argument(
        "--cep-threshold",
        type=float,
        default=None,
        help="有效定位的 CEP 门槛 [m]（默认取 replay meta，再缺省 500）",
    )
    aar_parser.add_argument(
        "--target",
        default=None,
        metavar="PLATFORM_ID",
        help="被定位的平台（默认取 localize objective）",
    )

    bench = subparsers.add_parser(
        "bench", help="跑 baseline 策略的评测，输出可比表格（T3b）"
    )
    bench.add_argument("--all", action="store_true", help="跑 scenarios/ 下所有想定")
    bench.add_argument(
        "--scenario", type=Path, action="append", default=None, help="指定想定（可重复）"
    )
    bench.add_argument("--episodes", type=int, default=5, help="每个组合跑几局（默认 5）")
    bench.add_argument(
        "--policies",
        default="random,always-jam,stealth",
        help="逗号分隔的策略名（默认 random,always-jam,stealth）",
    )
    bench.add_argument(
        "-o", "--out", type=Path, default=None, help="把结果写成 JSON"
    )
    bench.add_argument(
        "--markdown", type=Path, default=None, help="把结果写成 markdown 排行榜"
    )
    return parser


# --- bench：baseline 评测 -----------------------------------------------------


def _bench(args: argparse.Namespace) -> int:
    try:
        from .bench.policies import make_policy  # noqa: F401  (提前暴露 gymnasium 缺失)
        from .bench.runner import format_table, run_bench, write_json
    except ImportError as exc:  # pragma: no cover - 取决于环境
        print(
            f"[错误] bench 需要 gymnasium（{exc}）。装：pip install -e \".[bench]\"",
            file=sys.stderr,
        )
        return 1

    if args.all:
        paths = sorted(Path("scenarios").glob("*.yaml"))
    elif args.scenario:
        paths = list(args.scenario)
    else:
        print("[错误] 请给 --all 或 --scenario <path>", file=sys.stderr)
        return 2

    scenarios = []
    for path in paths:
        try:
            scenarios.append(load_scenario(path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1
    if not scenarios:
        print("[错误] 没有找到任何想定", file=sys.stderr)
        return 1

    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    try:
        rows = run_bench(scenarios, policy_names, episodes=args.episodes)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    print("SpectrumFog · baseline 评测")
    print(
        f"  {len(scenarios)} 个想定 × {len(policy_names)} 个策略 × "
        f"{args.episodes} 局（同一 seed 下各策略共用同一串随机数）"
    )
    print()
    print(format_table(rows))
    print()

    if args.out is not None:
        write_json(args.out, rows)
        print(f"  JSON → {args.out}")
    if args.markdown is not None:
        _write_markdown(args.markdown, rows)
        print(f"  markdown → {args.markdown}")
    print()
    print(_FOOTNOTE)
    return 0


def _write_markdown(path: Path, rows) -> None:
    """排行榜：按 (想定, 奖励) 排序的 markdown 表。"""
    lines = [
        "| 想定 | 策略 | 奖励（均值±标准差） | 干扰有效率 | 暴露率 | Pd 均值 | 首告警 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        warning = "—" if row.first_warning_s is None else f"{row.first_warning_s:.0f} s"
        lines.append(
            f"| {row.scenario} | `{row.policy}` | {row.reward_mean:.1f} ± "
            f"{row.reward_std:.0f} | {row.jammed_fraction * 100:.0f}% | "
            f"{row.exposure_fraction * 100:.0f}% | {row.pd_mean:.3f} | {warning} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "sim":
        return _sim(args)
    if args.command == "animate":
        return _animate(args)
    if args.command == "export-web":
        return _export_web(args)
    if args.command == "aar":
        return _aar(args)
    if args.command == "bench":
        return _bench(args)
    parser.error(f"未知子命令：{args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
