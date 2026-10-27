"""想定目录（5 个想定）都能加载、都能跑。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrumfog.engine.sim import Simulation
from spectrumfog.model.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [
    path
    for path in sorted((REPO_ROOT / "scenarios").glob("*.yaml"))
    if not path.stem.startswith("suite")
]


def test_catalog_has_at_least_five_scenarios():
    assert len(SCENARIOS) >= 5, [path.name for path in SCENARIOS]


@pytest.mark.parametrize("path", SCENARIOS, ids=[path.stem for path in SCENARIOS])
def test_scenario_runs_end_to_end(path):
    scenario = load_scenario(path)
    result = Simulation(scenario).run()

    assert result.meta["scenario"] == path.stem
    assert result.summary["n_ticks"] == int(
        round(scenario.meta.duration / scenario.meta.dt)
    ) + 1
    assert any(event["kind"] == "state" for event in result.events)
    assert any(event["kind"] == "link" for event in result.events)
    # 每个想定都要有 ESM 与目标，否则 AAR / bench 没法用
    assert scenario.sensors, f"{path.name} 缺传感器"
    assert result.meta["localize_target"], f"{path.name} 缺 localize 目标"
