"""Analytical formulas for tutorial theory-vs-simulation comparisons.

Reuses emsim.modes.rectangular where available; provides cavity, PEC, wavelength, etc.
Run from project root so that both emsim and Tutorial.common are importable.
"""

import numpy as np

from emsim.constants import C0, ETA0
from emsim.modes.rectangular import cutoff_frequency as _cutoff_frequency
from emsim.modes.rectangular import mode_impedance as _mode_impedance


def te10_cutoff(a: float) -> float:
    """TE10 cutoff frequency [Hz]: fc = c / (2*a)."""
    return C0 / (2.0 * a)


def cavity_frequency(m: int, n: int, p: int, a: float, b: float, d: float) -> float:
    """Rectangular cavity resonance [Hz]: f_mnp = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)."""
    return (C0 / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)


def te_impedance(m: int, n: int, a: float, b: float, f: float) -> complex:
    """TE mode impedance [Ohm]. Wrapper around emsim.modes.rectangular.mode_impedance."""
    return _mode_impedance(m, n, a, b, f)


def pec_reflection_coefficient() -> float:
    """PEC reflection coefficient: Gamma = -1 (total reflection, phase inverted)."""
    return -1.0


def free_space_impedance() -> float:
    """Free-space impedance [Ohm]: eta0 = sqrt(mu0/epsilon0) ~ 377 Ohm."""
    return float(ETA0)


def wavelength(f: float, eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Wavelength [m]: lambda = c / (f * sqrt(eps_r * mu_r))."""
    c = C0 / np.sqrt(eps_r * mu_r)
    return c / f


def measure_wave_speed(
    signal_history_1, signal_history_2, distance: float, dt: float
) -> float:
    """Measure wave propagation speed using cross-correlation.

    signal_1 is upstream, signal_2 downstream; the wave propagates in that direction.
    Returns measured speed [m/s].
    """
    s1 = np.asarray(signal_history_1, dtype=float)
    s2 = np.asarray(signal_history_2, dtype=float)
    corr = np.correlate(s1, s2, mode="full")
    positive_lag_slice = corr[len(s1) - 1 :]
    if len(positive_lag_slice) == 0:
        delay_steps = 1
    else:
        delay_steps = int(np.argmax(positive_lag_slice))
        if delay_steps < 1:
            delay_steps = 1
    delay_time = delay_steps * dt
    return distance / delay_time
