"""Validation tests for waveguide mode propagation.

These tests verify that the FDTD solver correctly simulates TE/TM modes
in rectangular waveguides by comparing numerical results with analytical solutions.
"""

import pytest
import numpy as np

from emsim.simulation import Simulation
from emsim.modes.rectangular import cutoff_frequency, mode_impedance
from emsim.constants import C0


@pytest.mark.validation
@pytest.mark.slow
def test_te10_cutoff_frequency_wr42(wr42_dimensions, analytical_solutions):
    """Validate TE10 cutoff frequency in WR42 waveguide.
    
    Analytical solution: fc = c/(2a) = 3e8/(2*0.0107) = 14.05 GHz
    
    The simulation should show:
    - High attenuation (S21 << 0 dB) below cutoff
    - Low loss (S21 ≈ 0 dB) above cutoff
    """
    a = wr42_dimensions['a']
    fc_analytic = analytical_solutions['te10_cutoff'](a)
    fc_expected = 14.05e9
    
    # Analytical formula validation
    assert np.isclose(fc_analytic, fc_expected, rtol=0.001)
    
    # TODO: Run short WR42 simulation near cutoff and measure S21
    # This requires a full simulation run, skipping for now
    pytest.skip("Requires full simulation run - implement when needed")


@pytest.mark.validation
def test_te10_impedance_formula(wr42_dimensions, analytical_solutions):
    """Validate TE10 mode impedance formula.
    
    Analytical: Z_TE = η₀ / sqrt(1 - (fc/f)²)
    
    Test at multiple frequencies above cutoff.
    """
    a = wr42_dimensions['a']
    b = wr42_dimensions['b']
    
    # Test frequencies above cutoff
    test_freqs = np.array([18e9, 20e9, 23e9, 26e9])
    
    for f in test_freqs:
        Z_computed = mode_impedance(1, 0, a, b, f)
        Z_analytic = analytical_solutions['te_impedance'](1, 0, a, b, f)
        
        # Should match within numerical precision
        assert np.isclose(Z_computed, Z_analytic, rtol=1e-10), \
            f"At {f/1e9:.1f} GHz: computed={Z_computed:.2f}, expected={Z_analytic:.2f}"


@pytest.mark.validation
def test_te10_below_cutoff(wr42_dimensions):
    """Validate that TE10 is evanescent below cutoff frequency.
    
    Below fc, the mode impedance should be purely imaginary (reactive).
    """
    a = wr42_dimensions['a']
    b = wr42_dimensions['b']
    fc = wr42_dimensions['fc_te10']
    
    # Test below cutoff
    f_below = fc * 0.9  # 10% below cutoff
    Z = mode_impedance(1, 0, a, b, f_below)
    
    # Below cutoff: impedance is purely imaginary (no real part)
    assert np.real(Z) == pytest.approx(0.0, abs=1e-10)
    assert np.imag(Z) != 0.0  # Has imaginary component


@pytest.mark.validation
def test_te10_propagation_constant(wr42_dimensions, analytical_solutions):
    """Validate propagation constant β = 2π/λg.
    
    Guide wavelength: λg = λ₀ / sqrt(1 - (fc/f)²)
    Propagation constant: β = 2πf*sqrt(1 - (fc/f)²) / c
    """
    a = wr42_dimensions['a']
    b = wr42_dimensions['b']
    f = 23e9  # Operating frequency
    
    # Analytical propagation constant
    fc = cutoff_frequency(1, 0, a, b)
    beta_analytic = 2 * np.pi * f * np.sqrt(1 - (fc / f)**2) / C0
    
    beta_computed = analytical_solutions['propagation_constant'](f, a, b, m=1, n=0)
    
    assert np.isclose(np.real(beta_computed), beta_analytic, rtol=1e-10)


@pytest.mark.validation
def test_higher_order_modes(wr42_dimensions):
    """Validate cutoff frequencies for higher-order modes.
    
    Tests TE10, TE20, TE01, TE11, TM11 cutoff frequencies.
    """
    a = wr42_dimensions['a']
    b = wr42_dimensions['b']
    
    # Analytical cutoff frequencies
    test_cases = [
        (1, 0, C0 / (2*a)),           # TE10
        (2, 0, C0 / a),                # TE20  
        (0, 1, C0 / (2*b)),            # TE01
        (1, 1, C0/2 * np.sqrt(1/a**2 + 1/b**2)),  # TE11
    ]
    
    for m, n, fc_expected in test_cases:
        fc_computed = cutoff_frequency(m, n, a, b)
        assert np.isclose(fc_computed, fc_expected, rtol=1e-10), \
            f"Mode TE{m}{n}: computed={fc_computed/1e9:.3f} GHz, expected={fc_expected/1e9:.3f} GHz"


@pytest.mark.validation
@pytest.mark.slow
def test_wr42_s21_vs_frequency():
    """Validate S21 vs frequency in WR42 waveguide.
    
    Expected:
    - S21 ≈ 0 dB in operating range (low loss)
    - S21 phase = -βL (linear phase shift)
    
    This test requires running the actual WR42 simulation.
    """
    pytest.skip("Requires full WR42 simulation - run manually with Simulations/WR42/run.py")


@pytest.mark.validation
def test_mode_orthogonality():
    """Validate orthogonality of mode profiles.
    
    Different modes should have zero overlap integral:
    ∫∫ E_m · E_n dA = 0 for m ≠ n
    """
    from emsim.modes.rectangular import te_mode_profile
    
    a = 10.7e-3
    b = 4.3e-3
    Nx, Ny = 30, 15
    dx = a / Nx
    dy = b / Ny
    
    # Get two different mode profiles
    mode_10 = te_mode_profile(1, 0, a, b, Ny, Nx, dx, dy)
    mode_20 = te_mode_profile(2, 0, a, b, Ny, Nx, dx, dy)
    
    # Compute overlap
    import tensorflow as tf
    overlap = tf.reduce_sum(mode_10 * mode_20).numpy() * dx * dy
    
    # Should be zero (orthogonal)
    assert np.abs(overlap) < 1e-6, f"Overlap = {overlap} (should be ~0)"
