"""ESM 观测模型：检出的物理、噪声缩放、虚假目标率、确定性。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectrumfog.engine import detect

SENSOR_POS = (0.0, 0.0)
EMITTER_POS = (30_000.0, 0.0)


def _observe(pos=(30_000.0, 0.0), power=90.0, seed=1, pfa=1e-6):
    return detect.observe(
        sensor_id="esm_1",
        sensor_pos=SENSOR_POS,
        sensor_bw_hz=1.0e6,
        sensor_noise_figure=10.0 ** 1.2,
        emitter_id="jammer_b",
        emitter_pos=pos,
        emitter_power_w=power,
        emitter_gain=1.0,
        emitter_freq_hz=3.0e9,
        t_s=12.0,
        pfa=pfa,
        rng=np.random.default_rng(seed),
    )


def test_strong_signal_is_detected_and_bearing_is_close():
    measurement = _observe()

    assert measurement.detected is True
    assert measurement.pd > 0.5
    assert measurement.aoa_rad == pytest.approx(0.0, abs=0.05)  # 目标在正东
    assert measurement.freq_hz == pytest.approx(3.0e9, abs=5.0e3)
    assert measurement.power_dbm < -60.0  # 30 km 处很弱
    assert measurement.toa_s is not None
    assert measurement.toa_s >= 12.0  # 传播时间只能让到达更晚


def test_weak_signal_expects_no_detection():
    """功率降到 1/1000 后 Pd 应该接近 0（不是"永远检不到"，而是概率极低）。"""
    measurement = _observe(power=0.09)
    assert measurement.pd < 0.01


def test_aoa_sigma_shrinks_with_snr():
    assert detect.aoa_sigma(1.0) > detect.aoa_sigma(100.0) > detect.aoa_sigma(10_000.0)
    # 上下夹紧，避免给出"完美"或"完全没用"的方差
    assert detect.aoa_sigma(10.0**12) >= math.radians(0.2)
    assert detect.aoa_sigma(1e-12) <= math.radians(20.0)


def test_measurement_is_reproducible_with_the_same_seed():
    first = _observe(seed=42)
    second = _observe(seed=42)
    assert first == second

    other = _observe(seed=43)
    assert (first.aoa_rad, first.draw) != (other.aoa_rad, other.draw)


def test_false_alarms_follow_the_configured_rate():
    rng = np.random.default_rng(11)
    alarms = [
        detect.false_alarm(
            sensor_id="esm_1",
            sensor_pos=SENSOR_POS,
            t_s=float(index),
            pfa_scan=1.0e-2,
            rng=rng,
        )
        for index in range(4000)
    ]
    hits = [alarm for alarm in alarms if alarm is not None]
    rate = len(hits) / len(alarms)

    assert 0.006 < rate < 0.014  # 1% ± 4σ
    assert all(hit.emitter_id is None for hit in hits)
    assert all(hit.detected for hit in hits)


def test_false_alarm_disabled_returns_none():
    assert (
        detect.false_alarm(
            sensor_id="esm_1",
            sensor_pos=SENSOR_POS,
            t_s=0.0,
            pfa_scan=0.0,
            rng=np.random.default_rng(1),
        )
        is None
    )


def test_measurement_event_is_self_contained():
    event = _observe().to_event()
    assert event["kind"] == "measurement"
    for key in ("sensor", "sensor_pos", "emitter", "t", "pd", "draw", "detected"):
        assert key in event
    assert event["sensor_pos"] == [0.0, 0.0]
