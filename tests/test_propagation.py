"""传播 / 链路预算的数值一致性（friis、FSPL、Eb/N0、抗干扰余量）。"""

from __future__ import annotations

import numpy as np
import pytest

from spectrumfog.physics import comms, propagation
from spectrumfog.units import SPEED_OF_LIGHT, db_to_linear


def test_free_space_path_loss_matches_definition():
    freq_hz = 3.0e9
    range_m = 1.0e3
    expected = (4.0 * np.pi * range_m / (SPEED_OF_LIGHT / freq_hz)) ** 2

    assert propagation.free_space_path_loss(
        freq_hz=freq_hz, range_m=range_m
    ) == pytest.approx(expected, rel=1.0e-12)
    assert propagation.free_space_path_loss_db(
        freq_hz=freq_hz, range_m=range_m
    ) == pytest.approx(10.0 * np.log10(expected), rel=1.0e-12)


def test_received_power_is_friis_and_obeys_r_squared():
    """Prx = Ptx*Gtx*Grx*(λ/4πd)^2；距离翻倍 → 功率除以 4。"""
    kwargs = dict(ptx_w=10.0, gtx=100.0, grx=2.0, freq_hz=3.0e9)
    prx_1km = propagation.received_power(range_m=1.0e3, **kwargs)
    prx_2km = propagation.received_power(range_m=2.0e3, **kwargs)

    lam = SPEED_OF_LIGHT / 3.0e9
    expected = 10.0 * 100.0 * 2.0 * (lam / (4.0 * np.pi * 1.0e3)) ** 2
    assert prx_1km == pytest.approx(expected, rel=1.0e-12)
    assert prx_2km == pytest.approx(prx_1km / 4.0, rel=1.0e-12)


def test_comms_link_power_equals_propagation():
    """链路预算与传播层必须是同一个式子，不许各写一套（§2 决策 3）。"""
    kwargs = dict(ptx_w=25.0, gtx=10.0 ** 1.5, grx=10.0**2.0, freq_hz=8.0e9)
    assert comms.link_received_power(range_m=12_000.0, **kwargs) == pytest.approx(
        propagation.received_power(range_m=12_000.0, **kwargs), rel=1.0e-15
    )


def test_noise_psd_is_k_t_f():
    assert comms.noise_psd(noise_figure=db_to_linear(4.0), temperature_k=290.0) == (
        pytest.approx(1.380649e-23 * 290.0 * db_to_linear(4.0), rel=1.0e-12)
    )


def test_ebn0_formula():
    """Eb/N0 = Prx / (N0 * Rb)，线性值。"""
    prx = 1.0e-15
    n0 = 4.0e-21
    rb = 1.0e6
    assert comms.ebn0(prx, n0, rb) == pytest.approx(prx / (n0 * rb), rel=1.0e-15)
    assert comms.ebn0_db(prx, n0, rb) == pytest.approx(
        10.0 * np.log10(prx / (n0 * rb)), rel=1.0e-15
    )


def test_aj_margin_and_link_availability():
    """AJ_margin = Gp + Mc - required_JS，> 0 视为链路可用。"""
    assert comms.aj_margin_db(20.0, 3.0, 10.0) == pytest.approx(13.0)
    assert bool(comms.link_available(20.0, 3.0, 10.0)) is True
    assert bool(comms.link_available(10.0, 0.0, 15.0)) is False


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        propagation.received_power(
            ptx_w=1.0, gtx=1.0, grx=1.0, freq_hz=3.0e9, range_m=0.0
        )
    with pytest.raises(ValueError):
        comms.ebn0(prx_w=1.0e-15, noise_psd_w_per_hz=4.0e-21, bit_rate_bps=0.0)
