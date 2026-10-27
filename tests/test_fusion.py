"""定位融合：几何、协方差、TDOA 与 AOA 的组合。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectrumfog.engine import fusion

TARGET = np.array([30_000.0, 0.0])
SENSORS = {
    "north": (0.0, 25_000.0),
    "south": (0.0, -25_000.0),
    "west": (-25_000.0, 0.0),
}


def bearing(sensor: tuple[float, float], point: np.ndarray) -> float:
    return math.atan2(point[1] - sensor[1], point[0] - sensor[0])


def test_exact_bearings_recover_the_point():
    measures = [
        fusion.Measure(sensor_id=name, sensor_pos=pos, aoa_rad=bearing(pos, TARGET))
        for name, pos in SENSORS.items()
    ]
    fix = fusion.solve_fix(measures)

    assert fix is not None
    assert fix.pos == pytest.approx(TARGET, abs=1e-6)
    assert fix.n_measurements == 3
    assert set(fix.sensors) == set(SENSORS)
    # 三站几何：协方差应该是有限的正定矩阵
    assert np.all(np.linalg.eigvalsh(fix.covariance) > 0.0)


def test_two_sensors_are_worse_than_three():
    """同样的测向噪声，三站比两站准（统计意义上的，用固定 seed）。"""
    rng = np.random.default_rng(7)
    sigma = math.radians(2.0)

    def noisy(names: tuple[str, ...]) -> fusion.Fix:
        measures = [
            fusion.Measure(
                sensor_id=name,
                sensor_pos=SENSORS[name],
                aoa_rad=bearing(SENSORS[name], TARGET) + float(rng.normal(0.0, sigma)),
                sigma_aoa_rad=sigma,
            )
            for name in names
        ]
        fix = fusion.solve_fix(measures)
        assert fix is not None
        return fix

    two = noisy(("north", "south"))
    three = noisy(("north", "south", "west"))
    assert three.gdop < two.gdop


def test_single_bearing_cannot_localise():
    measures = [
        fusion.Measure(
            sensor_id="north",
            sensor_pos=SENSORS["north"],
            aoa_rad=bearing(SENSORS["north"], TARGET),
        )
    ]
    assert fusion.solve_fix(measures) is None


def test_geometry_angle_is_reported():
    """南北两站看正东目标，两条视线的夹角 = 2·arctan(25/30) ≈ 79.6°。"""
    measures = [
        fusion.Measure(
            sensor_id=name,
            sensor_pos=SENSORS[name],
            aoa_rad=bearing(SENSORS[name], TARGET),
        )
        for name in ("north", "south")
    ]
    fix = fusion.solve_fix(measures)

    assert fix is not None
    assert fix.geometry_deg is not None
    expected = math.degrees(2.0 * abs(bearing(SENSORS["north"], TARGET)))
    assert fix.geometry_deg == pytest.approx(expected, rel=1e-6)
    assert fix.geometry_deg == pytest.approx(79.611, rel=1e-3)


def test_tdoa_alone_is_very_precise():
    """三个站的到达时间差足以单独定位（σ_toa = 100 ns → 30 m）。"""
    sigma_toa = 1.0e-7
    measures = [
        fusion.Measure(
            sensor_id=name,
            sensor_pos=pos,
            toa_s=0.0 + math.dist(pos, TARGET) / fusion.SPEED_OF_LIGHT_MPS,
            sigma_toa_s=sigma_toa,
        )
        for name, pos in SENSORS.items()
    ]
    fix = fusion.solve_fix(measures)

    assert fix is not None
    assert math.dist(fix.pos, TARGET) < 200.0
    assert fix.n_measurements == 3


def test_bearing_intersection_handles_parallel_lines():
    # 两条平行线（都朝正东）
    assert fusion.bearing_intersection((0, 0), 0.0, (0, 1000), 0.0) is None
    # 从 (0,0) 朝东的线，与从 (1000,1000) 朝南的线，交于 (1000, 0)
    point = fusion.bearing_intersection((0, 0), 0.0, (1000, 1000), -math.pi / 2)
    assert point == pytest.approx([1000.0, 0.0], abs=1e-6)


def test_covariance_grows_with_bearing_noise():
    def solve(sigma_deg: float) -> float:
        measures = [
            fusion.Measure(
                sensor_id=name,
                sensor_pos=pos,
                aoa_rad=bearing(pos, TARGET),
                sigma_aoa_rad=math.radians(sigma_deg),
            )
            for name, pos in SENSORS.items()
        ]
        fix = fusion.solve_fix(measures)
        assert fix is not None
        return fix.gdop

    assert solve(4.0) > solve(1.0)
