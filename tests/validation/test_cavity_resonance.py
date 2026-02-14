"""Validation tests for cavity resonance frequencies.

These tests verify that the FDTD solver correctly computes resonant frequencies
of closed cavities (PEC walls on all sides) by comparing with analytical solutions.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.boundaries.pec import apply_pec
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import C0


@pytest.mark.validation
@pytest.mark.slow
def test_rectangular_cavity_modes(analytical_solutions):
    """Validate resonance frequencies of rectangular cavity with PEC walls.
    
    Analytical solution: f_mnp = (c/2) * sqrt((m/a)² + (n/b)² + (p/d)²)
    
    Test modes: (1,0,1), (1,1,0), (0,1,1), (2,0,1)
    """
    # Cavity dimensions
    a = 10e-3  # x-direction
    b = 8e-3   # y-direction
    d = 6e-3   # z-direction
    
    # Create grid
    grid = YeeGrid(
        x_range=(0, a),
        y_range=(0, b),
        z_range=(0, d),
        f0=20e9,  # Center frequency
        resolution=20,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0, mu_r=1.0, sigma=0.0)
    mat.compute_coefficients(grid.dt)
    
    # Initialize fields
    Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    
    # Gaussian pulse to excite multiple modes
    source = GaussianPulse(f0=20e9, bandwidth=15e9)
    
    # Source in center
    i_src = grid.Nz // 2
    j_src = grid.Ny // 2
    k_src = grid.Nx // 2
    
    # Record E-field at center
    Ez_record = []
    
    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    n_steps = 3000
    for n in range(n_steps):
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        
        # Apply PEC on all 6 faces
        apply_pec(Ex, Ey, Ez, {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'})
        
        # Inject pulse (only first ~100 steps); use tensor_scatter_nd_update + assign
        if n < 100:
            amplitude = source(n * grid.dt)
            amp = float(amplitude.numpy()) if hasattr(amplitude, 'numpy') else float(amplitude)
            idx = tf.constant([[i_src, j_src, k_src]], dtype=tf.int32)
            new_val = Ez[i_src, j_src, k_src].numpy() + amp
            updated = tf.tensor_scatter_nd_update(Ez.read_value(), idx, tf.constant([new_val], dtype=Ez.dtype))
            Ez.assign(updated)
        
        # Record
        Ez_record.append(Ez[i_src, j_src, k_src].numpy())
    
    Ez_record = np.array(Ez_record)
    
    # FFT to find resonance peaks
    from scipy.fft import fft, fftfreq
    N = len(Ez_record)
    fft_vals = fft(Ez_record)
    freqs = fftfreq(N, grid.dt)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask])
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag) * 0.1)
    resonance_freqs = freqs_pos[peaks]
    
    print(f"\nFound {len(resonance_freqs)} resonance peaks:")
    for f_res in sorted(resonance_freqs):
        print(f"  {f_res/1e9:.3f} GHz")
    
    # Compute analytical resonances
    test_modes = [
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (2, 0, 1),
        (1, 1, 1),
    ]
    
    print("\nAnalytical resonances:")
    analytical_freqs = []
    for m, n, p in test_modes:
        f_analytical = analytical_solutions['cavity_frequency'](m, n, p, a, b, d)
        analytical_freqs.append(f_analytical)
        print(f"  Mode ({m},{n},{p}): {f_analytical/1e9:.3f} GHz")
    
    # Check that at least some analytical frequencies are found
    # (tolerance: ±10% due to numerical dispersion and coarse grid)
    matches = 0
    for f_ana in analytical_freqs:
        for f_sim in resonance_freqs:
            if np.abs(f_sim - f_ana) / f_ana < 0.10:
                matches += 1
                break
    
    print(f"\nMatches: {matches}/{len(analytical_freqs)}")
    
    # At least 2 modes should match
    assert matches >= 2, f"Only {matches} cavity modes matched (expected >= 2)"


@pytest.mark.validation
def test_cavity_fundamental_mode(analytical_solutions):
    """Test the fundamental (lowest) mode of a rectangular cavity.
    
    For a×b×d cavity with a > b > d, the fundamental mode is usually (1,0,1) or (1,1,0).
    """
    a = 12e-3
    b = 8e-3
    d = 6e-3
    
    # Possible fundamental modes
    candidates = [
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
    ]
    
    f_min = float('inf')
    mode_min = None
    
    for m, n, p in candidates:
        if m == 0 and n == 0 and p == 0:
            continue  # Skip (0,0,0)
        
        f = analytical_solutions['cavity_frequency'](m, n, p, a, b, d)
        if f < f_min:
            f_min = f
            mode_min = (m, n, p)
    
    print(f"Fundamental mode: {mode_min} at {f_min/1e9:.3f} GHz")
    
    # Verify it's a valid frequency
    assert f_min > 0
    assert f_min < 100e9  # Reasonable range


@pytest.mark.validation
def test_cavity_quality_factor():
    """Test quality factor Q of a lossless cavity.
    
    For a lossless cavity, Q → ∞ (resonances are infinitely sharp).
    In practice, limited by numerical damping and grid resolution.
    """
    pytest.skip("Quality factor test - requires long simulation and spectral analysis")


@pytest.mark.validation
def test_cubic_cavity_degeneracy(analytical_solutions):
    """Test mode degeneracy in a cubic cavity.
    
    In a cubic cavity (a=b=d), certain modes have the same frequency due to symmetry.
    For example: (1,0,1), (0,1,1), (1,1,0) all have the same frequency.
    """
    a = 10e-3
    
    # Degenerate modes in cubic cavity
    mode_101 = analytical_solutions['cavity_frequency'](1, 0, 1, a, a, a)
    mode_011 = analytical_solutions['cavity_frequency'](0, 1, 1, a, a, a)
    mode_110 = analytical_solutions['cavity_frequency'](1, 1, 0, a, a, a)
    
    print(f"Mode (1,0,1): {mode_101/1e9:.6f} GHz")
    print(f"Mode (0,1,1): {mode_011/1e9:.6f} GHz")
    print(f"Mode (1,1,0): {mode_110/1e9:.6f} GHz")
    
    # Should all be equal (degenerate)
    assert np.isclose(mode_101, mode_011, rtol=1e-10)
    assert np.isclose(mode_101, mode_110, rtol=1e-10)
    assert np.isclose(mode_011, mode_110, rtol=1e-10)


@pytest.mark.validation
def test_cavity_vs_waveguide_cutoff():
    """Compare cavity resonance with waveguide cutoff.
    
    A cavity with length L in z-direction has modes (m,n,p).
    For p=0, the frequency equals the TE_mn cutoff of a waveguide with same cross-section.
    """
    a = 10.7e-3
    b = 4.3e-3
    d = 50e-3  # Length doesn't matter for p=0
    
    # Cavity mode (1,0,0) - no variation in z
    from emsim.modes.rectangular import cutoff_frequency
    fc_te10 = cutoff_frequency(1, 0, a, b)
    
    # This should equal cavity (1,0,0) mode
    # But (1,0,0) is not a valid cavity mode (needs p > 0 for standing wave in z)
    # Actually, for p=0, cavity acts like infinite waveguide
    
    # Instead, compare (1,0,1) cavity with TE10 cutoff
    from emsim.constants import C0
    f_cavity_101 = C0 / 2 * np.sqrt((1/a)**2 + (1/d)**2)
    
    print(f"TE10 cutoff: {fc_te10/1e9:.3f} GHz")
    print(f"Cavity (1,0,1): {f_cavity_101/1e9:.3f} GHz")
    
    # Cavity frequency should be higher than cutoff (due to z-component)
    assert f_cavity_101 > fc_te10


@pytest.mark.validation
def test_analytical_formula_consistency(analytical_solutions):
    """Test internal consistency of analytical cavity formula."""
    a, b, d = 10e-3, 8e-3, 6e-3
    
    # Manually compute
    f_manual = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2 + (1/d)**2)
    
    # Use fixture
    f_fixture = analytical_solutions['cavity_frequency'](1, 1, 1, a, b, d)
    
    assert np.isclose(f_manual, f_fixture, rtol=1e-10)


@pytest.mark.validation
@pytest.mark.slow
def test_cavity_field_distribution():
    """Test that field distribution matches mode shape.
    
    Mode (1,0,1): E_y ~ sin(πx/a) * sin(πz/d)
    """
    pytest.skip("Field distribution test - requires detailed spatial sampling")
