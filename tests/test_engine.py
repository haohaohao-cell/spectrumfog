"""固定步长时钟与确定性主循环。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrumfog.engine.clock import Clock
from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario, parse_scenario
from spectrumfog.replay import write_replay

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDOFF_YAML = REPO_ROOT / "scenarios" / "02_standoff.yaml"
DUEL_YAML = REPO_ROOT / "scenarios" / "01_duel.yaml"


# --- Clock -------------------------------------------------------------------


def test_clock_covers_closed_interval_in_integer_steps():
    """t = 0, dt, 2dt, ..., duration，时间由 index*dt 算出，不累加。"""
    clock = Clock(dt=2.0, duration=240.0)

    assert clock.n_ticks == 121
    assert len(clock) == 121
    assert clock.n_intervals == 120

    ticks = list(clock)
    assert ticks[0].t == 0.0
    assert ticks[-1].t == 240.0
    assert all(tick.dt == 2.0 for tick in ticks)
    assert [tick.index for tick in ticks] == list(range(121))
    # 严格单调、无漂移
    times = [tick.t for tick in ticks]
    assert times == [index * 2.0 for index in range(121)]


def test_clock_timestamp_is_exact_not_accumulated():
    """1000 步累加 0.1 会漂；index*dt 不会。"""
    clock = Clock(dt=0.1, duration=100.0)
    ticks = list(clock)
    assert ticks[1000].t == pytest.approx(100.0, abs=0.0)
    assert ticks[3].t == 0.30000000000000004  # 3*0.1 的二进制表示，但仍是"算出来的"
    assert ticks[10].t == 1.0


def test_clock_rejects_non_positive_steps():
    with pytest.raises(ValueError):
        Clock(dt=0.0, duration=10.0)
    with pytest.raises(ValueError):
        Clock(dt=1.0, duration=0.0)
    with pytest.raises(ValueError):
        Clock(dt=float("nan"), duration=10.0)
    with pytest.raises(ValueError):
        Clock(dt=1.0, duration=float("inf"))


# --- Simulation --------------------------------------------------------------


def test_standoff_run_is_deterministic():
    """同一想定 + 同一 seed → 事件流逐条相同，写出的字节也相同。"""
    scenario = load_scenario(STANDOFF_YAML)
    first = Simulation(scenario).run()
    second = Simulation(scenario).run()

    assert first.events == second.events
    assert first.summary == second.summary

    tmp = Path(__file__).parent / "_determinism_a.jsonl"
    other = Path(__file__).parent / "_determinism_b.jsonl"
    try:
        write_replay(tmp, first.events)
        write_replay(other, second.events)
        assert tmp.read_bytes() == other.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)
        other.unlink(missing_ok=True)


def test_seed_changes_sampling_but_not_physics():
    """换 seed 只该改变抽样结果（掷点 / detected），物理量必须一模一样。"""
    scenario = load_scenario(STANDOFF_YAML)
    base = load_scenario(STANDOFF_YAML)
    assert base.meta.seed == 7

    run_a = Simulation(scenario).run()
    # 用同一份想定但换 seed 的等价构造（frozen 模型，走 model_copy）
    reseeded = scenario.model_copy(update={"meta": scenario.meta.model_copy(update={"seed": 8})})
    run_b = Simulation(reseeded).run()

    links_a = [event for event in run_a.events if event["kind"] == "link"]
    links_b = [event for event in run_b.events if event["kind"] == "link"]
    for left, right in zip(links_a, links_b):
        assert left["pd"] == right["pd"]
        assert left["js"] == right["js"]

    draws_a = [event["draw"] for event in run_a.events if event["kind"] == "detection"]
    draws_b = [event["draw"] for event in run_b.events if event["kind"] == "detection"]
    assert draws_a != draws_b


def test_standoff_physics_is_monotone_in_time():
    """干扰机一路逼近：J/S 单调升、Pd 单调降、两条半径单调缩。"""
    scenario = load_scenario(STANDOFF_YAML)
    result = Simulation(scenario).run()
    links = [event for event in result.events if event["kind"] == "link"]

    js = [event["js"] for event in links]
    pd = [event["pd"] for event in links]
    burn = [event["burn_through_m"] for event in links]
    rings = [
        event["detection_ring_m"] for event in links if event["detection_ring_m"] is not None
    ]

    assert all(later > earlier for earlier, later in zip(js, js[1:]))
    assert all(later < earlier for earlier, later in zip(pd, pd[1:]))
    assert all(later < earlier for earlier, later in zip(burn, burn[1:]))
    assert all(later < earlier for earlier, later in zip(rings, rings[1:]))


def test_standoff_summary_matches_the_story():
    """T1 的验收线：干扰机从 60 km 压到 4 km，目标在片中段丢失。"""
    scenario = load_scenario(STANDOFF_YAML)
    summary = Simulation(scenario).run().summary

    assert summary["n_ticks"] == 121
    assert summary["duration_s"] == 240.0
    assert summary["jammer_range_start_m"] == pytest.approx(60_000.0)
    assert summary["jammer_range_end_m"] == pytest.approx(4_000.0)

    # Pd 跌破门槛发生在 60%–80% 之间（这是动画的"高潮"位置，故意调到这里）
    flip = summary["pd_below_threshold_at_s"]
    assert flip is not None
    assert 0.55 * 240.0 <= flip <= 0.80 * 240.0

    assert summary["pd_max"] > 0.9
    assert summary["pd_min"] < 1.0e-3
    # 单次扫描命中率是抽样结果，和"是否丢失"不是一回事
    assert 0.0 < summary["scan_detection_fraction"] < 1.0


def test_target_falls_back_to_jammer_without_detect_objective():
    """01_duel 没有 type: detect 的 objective → 自屏蔽，目标就是干扰机。"""
    scenario = load_scenario(DUEL_YAML)
    result = Simulation(scenario).run()
    assert result.meta["target"] == "jammer_b"
    links = [event for event in result.events if event["kind"] == "link"]
    assert links[0]["range_target_m"] == pytest.approx(links[0]["range_jammer_m"])


def test_meta_separates_detect_target_from_localize_target():
    """`target` 是雷达要探测的，`localize_target` 是被 ESM 定位的——两者不能混。"""
    scenario = load_scenario(STANDOFF_YAML)
    result = Simulation(scenario).run()

    assert result.meta["target"] == "target_c"
    assert result.meta["localize_target"] == "jammer_b"
    assert result.meta["cep_threshold_m"] == pytest.approx(500.0)


def test_state_events_carry_positions_and_emitters():
    scenario = load_scenario(STANDOFF_YAML)
    result = Simulation(scenario).run()
    states = [event for event in result.events if event["kind"] == "state"]

    assert len(states) == 3 * 121  # 3 个平台 × 121 帧
    first_radar = next(state for state in states if state["platform"] == "radar_a")
    assert first_radar["pos"] == [0.0, 0.0]
    assert first_radar["emitter"] == "radar"
    assert first_radar["on"] is True

    first_target = next(state for state in states if state["platform"] == "target_c")
    assert first_target["emitter"] is None
    assert first_target["on"] is False

    last_jammer = [
        state for state in states if state["platform"] == "jammer_b"
    ][-1]
    assert last_jammer["pos"] == pytest.approx([4000.0, 0.0])


def test_simulation_rejects_bad_probabilities():
    scenario = load_scenario(STANDOFF_YAML)
    with pytest.raises(ValueError):
        Simulation(scenario, pfa=0.0)
    with pytest.raises(ValueError):
        Simulation(scenario, pfa=1.0)
    with pytest.raises(ValueError):
        Simulation(scenario, pd_threshold=-0.1)


def test_simulation_honours_short_clock():
    """短想定应该只跑这么几步：duration / dt 决定一切。"""
    scenario = parse_scenario(
        {
            "meta": {"name": "short", "seed": 1, "dt": 5.0, "duration": 20.0},
            "platforms": [
                {
                    "id": "r",
                    "side": "blue",
                    "pos": [0, 0],
                    "emitters": [
                        {
                            "kind": "radar",
                            "freq_hz": 3.0e9,
                            "bw_hz": 2.0e6,
                            "pt_w": 1.0e5,
                            "gain_db": 30,
                            "sigma_m2": 1.0,
                        }
                    ],
                },
                {
                    "id": "j",
                    "side": "red",
                    "pos": [20_000, 0],
                    "emitters": [
                        {
                            "kind": "noise_jammer",
                            "freq_hz": 3.0e9,
                            "bw_hz": 5.0e7,
                            "pj_w": 50.0,
                            "gain_db": 3,
                        }
                    ],
                },
            ],
        }
    )
    result = Simulation(scenario).run()
    assert result.summary["n_ticks"] == 5  # 0,5,10,15,20
    assert result.summary["duration_s"] == 20.0
    assert len(result.events) == 1 + 5 * (2 + 1 + 1)  # meta + 每帧(main 2 平台 + link + detection)
