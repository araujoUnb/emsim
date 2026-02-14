"""Shared fixtures for pytest across all test modules.

This module provides reusable fixtures for common objects like grids,
sources, and analytical solutions for validation tests.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import C0, EPS0, MU0, ETA0


@pytest.fixture
def small_grid():
    """Small grid for fast unit tests.
    
    Returns a 10.7mm × 4.3mm × 20mm grid at 10 GHz with λ/10 resolution.
    Total cells: ~25 × 10 × 50 = 12,500 (fast for testing).
    """
    return YeeGrid(
        x_range=(0, 10.7e-3),
        y_range=(0, 4.3e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=10,
        courant=0.5
    )


@pytest.fixture
def medium_grid():
    """Medium grid for integration tests.
    
    Returns a WR42-sized grid at 23 GHz with λ/20 resolution.
    """
    return YeeGrid(
        x_range=(0, 10.7e-3),
        y_range=(0, 4.3e-3),
        z_range=(0, 50e-3),
        f0=23e9,
        resolution=20,
        courant=0.5
    )


@pytest.fixture
def gaussian_source():
    """Standard Gaussian pulse source at 10 GHz."""
    return GaussianPulse(f0=10e9, bandwidth=5e9)


@pytest.fixture
def gaussian_source_24ghz():
    """Gaussian pulse for 2.4 GHz antenna simulations."""
    return GaussianPulse(f0=2.4e9, bandwidth=1e9)


@pytest.fixture
def analytical_solutions():
    """Collection of analytical solutions for validation tests.
    
    Returns a dictionary of functions for computing exact solutions:
    - te10_cutoff: TE10 cutoff frequency in rectangular waveguide
    - te_impedance: TE mode impedance
    - cavity_frequency: Rectangular cavity resonance frequencies
    - plane_wave_reflection_pec: PEC reflection coefficient (= -1)
    """
    def te10_cutoff(a):
        """TE10 cutoff frequency: fc = c/(2a)."""
        return C0 / (2 * a)
    
    def te_impedance(m, n, a, b, f):
        """TE mode impedance: Z_TE = η / sqrt(1 - (fc/f)²)."""
        from emsim.modes.rectangular import cutoff_frequency
        fc = cutoff_frequency(m, n, a, b)
        if f <= fc:
            return 0.0  # Below cutoff (evanescent)
        return ETA0 / np.sqrt(1 - (fc / f)**2)
    
    def cavity_frequency(m, n, p, a, b, d):
        """Rectangular cavity resonance: f_mnp = (c/2)*sqrt((m/a)² + (n/b)² + (p/d)²)."""
        return (C0 / 2) * np.sqrt((m / a)**2 + (n / b)**2 + (p / d)**2)
    
    def plane_wave_reflection_pec():
        """PEC reflection coefficient: Γ = -1 (total reflection with phase inversion)."""
        return -1.0
    
    def free_space_impedance():
        """Free space impedance: η₀ = sqrt(μ₀/ε₀) = 377 Ω."""
        return ETA0
    
    def wavelength(f, eps_r=1.0, mu_r=1.0):
        """Wavelength: λ = c / (f * sqrt(εᵣ * μᵣ))."""
        c = C0 / np.sqrt(eps_r * mu_r)
        return c / f
    
    def propagation_constant(f, a, b, m=1, n=0):
        """Propagation constant β in waveguide."""
        from emsim.modes.rectangular import cutoff_frequency
        fc = cutoff_frequency(m, n, a, b)
        if f <= fc:
            # Evanescent: β is imaginary
            return 0.0 + 1j * 2 * np.pi * fc * np.sqrt((fc/f)**2 - 1) / C0
        else:
            # Propagating
            return 2 * np.pi * f * np.sqrt(1 - (fc/f)**2) / C0
    
    return {
        'te10_cutoff': te10_cutoff,
        'te_impedance': te_impedance,
        'cavity_frequency': cavity_frequency,
        'plane_wave_reflection_pec': plane_wave_reflection_pec,
        'free_space_impedance': free_space_impedance,
        'wavelength': wavelength,
        'propagation_constant': propagation_constant,
    }


@pytest.fixture
def wr42_dimensions():
    """Standard WR42 waveguide dimensions."""
    return {
        'a': 10.7e-3,  # Width [m]
        'b': 4.3e-3,   # Height [m]
        'fc_te10': 14.05e9,  # TE10 cutoff [Hz]
        'operating_range': (18e9, 26.5e9),  # Typical operating range [Hz]
    }


@pytest.fixture
def patch_antenna_params():
    """Standard 2.4 GHz patch antenna parameters."""
    return {
        'freq': 2.4e9,
        'patch_width': 32e-3,
        'patch_length': 40e-3,
        'substrate_thickness': 1.524e-3,
        'substrate_eps_r': 3.38,
        'feed_resistance': 50.0,
        'expected_resonance': 2.4e9,
        'tolerance': 0.05,  # ±5%
    }


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (validation tests)")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "validation: marks physical validation tests")
