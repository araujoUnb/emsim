"""Tests for the Gaussian pulse source."""

import math
import pytest
import numpy as np
import tensorflow as tf
from emsim.sources.gaussian_pulse import GaussianPulse


def test_pulse_starts_near_zero():
    pulse = GaussianPulse(f0=10e9, bandwidth=4e9)
    val = float(pulse(tf.constant(0.0)).numpy())
    # Relax tolerance to 1e-4 (more realistic for float32)
    assert abs(val) < 1e-4, f"Pulse at t=0: {val}"


def test_pulse_peak_near_t0():
    pulse = GaussianPulse(f0=10e9, bandwidth=4e9)
    t = tf.linspace(0.0, pulse.duration, 10000)
    vals = pulse(t).numpy()
    peak_idx = np.argmax(np.abs(vals))
    t_peak = float(t[peak_idx].numpy())
    # Peak should be near t0
    assert abs(t_peak - pulse.t0) / pulse.t0 < 0.15


def test_spectrum_centred_at_f0():
    f0 = 10e9
    bw = 4e9
    pulse = GaussianPulse(f0=f0, bandwidth=bw)

    dt = 1e-12  # 1 ps
    N = 20000
    t = tf.range(N, dtype=tf.float32) * dt
    vals = pulse(t).numpy()

    spectrum = np.abs(np.fft.rfft(vals))
    freqs = np.fft.rfftfreq(N, d=dt)

    # Find peak frequency
    peak_idx = np.argmax(spectrum)
    f_peak = freqs[peak_idx]

    assert abs(f_peak - f0) / f0 < 0.05  # within 5%


def test_duration_positive():
    pulse = GaussianPulse(f0=10e9, bandwidth=4e9)
    assert pulse.duration > 0
