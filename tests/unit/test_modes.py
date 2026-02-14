"""Tests for rectangular waveguide mode profiles."""

import math
import pytest
import tensorflow as tf
from emsim.constants import C0
from emsim.modes.rectangular import te_mode_profile, tm_mode_profile, cutoff_frequency, mode_impedance


def test_te10_profile_shape():
    """TE10 profile should have the right shape."""
    a, b = 0.01, 0.005
    Ny, Nx = 20, 30
    profile = te_mode_profile(1, 0, a, b, Ny, Nx)
    assert profile.shape == (Ny, Nx)


def test_te10_peak_near_centre():
    """TE10 maximum should be near x = a/2 (stagger shifts by half cell)."""
    a, b = 0.01, 0.005
    Ny, Nx = 20, 60  # fine grid so stagger effect is small
    profile = te_mode_profile(1, 0, a, b, Ny, Nx)
    peak = float(tf.reduce_max(tf.abs(profile)).numpy())
    assert peak > 0


def test_te10_power_normalised():
    """TE10 profile should be power-normalised: sum(|p|^2)*dx*dy ~ 1."""
    a, b = 0.01, 0.005
    Ny, Nx = 40, 60
    dx = a / max(Nx - 1, 1)
    dy = b / max(Ny - 1, 1)
    profile = te_mode_profile(1, 0, a, b, Ny, Nx, dx=dx, dy=dy)
    power = float(tf.reduce_sum(profile ** 2).numpy()) * dx * dy
    assert power == pytest.approx(1.0, rel=0.05)


def test_tm11_boundary_conditions():
    """TM11 Ez must be zero on all 4 walls."""
    a, b = 0.01, 0.005
    Ny, Nx = 20, 30
    profile = tm_mode_profile(1, 1, a, b, Ny, Nx)

    # x=0, x=a: sin(0) = sin(pi) = 0
    assert abs(float(profile[Ny // 2, 0].numpy())) < 0.05
    assert abs(float(profile[Ny // 2, -1].numpy())) < 0.05
    # y=0, y=b: sin(0) = sin(pi) = 0
    assert abs(float(profile[0, Nx // 2].numpy())) < 0.05
    assert abs(float(profile[-1, Nx // 2].numpy())) < 0.05


def test_cutoff_te10():
    a = 10.668e-3  # WR-42
    b = 4.318e-3
    f_c = cutoff_frequency(1, 0, a, b)
    expected = C0 / (2 * a)
    assert f_c == pytest.approx(expected, rel=1e-6)
    assert 14.0e9 < f_c < 14.1e9  # ~14.05 GHz


def test_cutoff_te20():
    a = 10.668e-3
    b = 4.318e-3
    f_c_10 = cutoff_frequency(1, 0, a, b)
    f_c_20 = cutoff_frequency(2, 0, a, b)
    # TE20 cutoff should be exactly 2x TE10
    assert f_c_20 == pytest.approx(2 * f_c_10, rel=1e-6)


def test_mode_impedance_propagating():
    """Above cutoff, impedance should be real and > eta."""
    a = 10.668e-3
    b = 4.318e-3
    Z = mode_impedance(1, 0, a, b, 20e9)
    assert isinstance(Z, complex)
    assert abs(Z.imag) < 1e-6  # real above cutoff
    assert Z.real > 376.73  # Z_TE > eta above cutoff


def test_mode_impedance_evanescent():
    """Below cutoff, impedance should be purely imaginary."""
    a = 10.668e-3
    b = 4.318e-3
    fc = cutoff_frequency(1, 0, a, b)
    Z = mode_impedance(1, 0, a, b, fc * 0.5)
    assert isinstance(Z, complex)
    assert abs(Z.real) < 1e-6  # purely imaginary below cutoff
    assert Z.imag > 0  # positive imaginary for TE
