"""指标口径：CEP 分位数、协方差反推、暴露窗口。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectrumfog.metrics import (
    cep_from_covariance,
    cep_from_samples,
    exposure_windows,
    t_expose,
)


def test_cep_is_a_sample_quantile_not_an_approximation():
    errors = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    assert cep_from_samples(errors, 0.5) == pytest.approx(300.0)
    assert cep_from_samples(errors, 0.9) == pytest.approx(460.0)


def test_cep_ignores_non_finite_samples_and_rejects_empty():
    assert cep_from_samples(np.array([100.0, np.nan, 300.0]), 0.5) == pytest.approx(200.0)
    with pytest.raises(ValueError):
        cep_from_samples(np.array([]))
    with pytest.raises(ValueError):
        cep_from_samples(np.array([1.0]), 1.5)


def test_covariance_cep_matches_the_rayleigh_constant():
    """各向同性 σ=100 m 时，CEP50 ≈ 1.177σ、CEP90 ≈ 2.146σ。"""
    cep50, cep90 = cep_from_covariance(np.diag([100.0**2, 100.0**2]), samples=20000)
    assert cep50 == pytest.approx(1.1774 * 100.0, rel=0.02)
    assert cep90 == pytest.approx(2.1460 * 100.0, rel=0.02)


def test_covariance_cep_scales_with_the_ellipse():
    """σ 放大 4 倍 → CEP 放大 4 倍（注意 diag 里是方差）。"""
    cep50, _ = cep_from_covariance(np.diag([400.0**2, 100.0**2]), samples=20000)
    reference, _ = cep_from_covariance(np.diag([100.0**2, 25.0**2]), samples=20000)
    assert cep50 == pytest.approx(4.0 * reference, rel=0.05)


def test_covariance_cep_rejects_bad_input():
    with pytest.raises(ValueError):
        cep_from_covariance(np.eye(3))
    with pytest.raises(ValueError):
        cep_from_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_exposure_windows_split_and_sum():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ceps = np.array([900.0, 400.0, 300.0, 800.0, 450.0, 100.0])

    windows = exposure_windows(times, ceps, threshold_m=500.0)
    assert len(windows) == 2
    # 每个 tick 代表 [t, t+dt)
    assert (windows[0].start_s, windows[0].end_s) == (1.0, 3.0)
    assert (windows[1].start_s, windows[1].end_s) == (4.0, 6.0)
    assert windows[0].peak_cep_m == pytest.approx(400.0)
    assert t_expose(times, ceps, 500.0) == pytest.approx(4.0)


def test_single_frame_exposure_counts_as_one_dt():
    """只暴露一帧也要算 dt，不能被算成 0 秒。"""
    assert t_expose(np.array([5.0]), np.array([100.0]), 500.0, dt=2.0) == pytest.approx(2.0)


def test_exposure_respects_dt():
    times = np.array([0.0, 2.0, 4.0])
    ceps = np.array([100.0, 100.0, 900.0])
    assert t_expose(times, ceps, 500.0, dt=2.0) == pytest.approx(4.0)


def test_exposure_is_zero_when_never_inside_threshold():
    assert t_expose(np.array([0.0, 1.0]), np.array([900.0, 800.0]), 500.0) == 0.0


def test_cep_is_monotone_in_quantile():
    errors = np.linspace(1.0, 1000.0, 101)
    assert cep_from_samples(errors, 0.9) > cep_from_samples(errors, 0.5)
    assert math.isclose(cep_from_samples(errors, 0.5), 500.5, rel_tol=1e-9)
