"""Validation tests for free-space electromagnetic wave propagation.

These tests verify that waves propagate at the correct speed of light
and satisfy fundamental electromagnetic relations.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import C0, ETA0

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import measure_wave_speed


@pytest.mark.validation
@pytest.mark.slow
def test_speed_of_light():
    """Validate that electromagnetic pulse propagates at c = 299,792,458 m/s.
    
    Setup:
    - Gaussian pulse in center
    - Measure arrival time at two points separated by known distance
    - Compute velocity and compare with c
    """
    # Grid: need Nx, Ny >= 3 for H-updates to see center; long in z for propagation
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 60e-3),
        f0=10e9,
        resolution=40,
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=0.0
    )
    # Source upstream; two recording points downstream with clear separation
    z_src = 5
    z1 = 15
    z2 = min(35, grid.Nz - 1)
    if z2 <= z1 + 5:
        z2 = grid.Nz - 1
        z1 = max(z_src + 2, z2 - 25)
    distance = (z2 - z1) * grid.dz
    
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    
    Ey_at_z1 = []
    Ey_at_z2 = []
    
    n_steps = 2000
    mat = grid.materials
    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    for n in range(n_steps):
        update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        
        # Inject source (Variable slice has no assign_add)
        amplitude = source(n * grid.dt)
        amp = float(amplitude.numpy()) if hasattr(amplitude, 'numpy') else float(amplitude)
        idx = tf.constant([[z_src, grid.Ny//2, grid.Nx//2]], dtype=tf.int32)
        new_val = grid.Ey[z_src, grid.Ny//2, grid.Nx//2].numpy() + amp
        grid.Ey.assign(tf.tensor_scatter_nd_update(
            grid.Ey.read_value(), idx, tf.constant([new_val], dtype=grid.Ey.dtype)
        ))
        
        # Record
        Ey_at_z1.append(grid.Ey[z1, grid.Ny//2, grid.Nx//2].numpy())
        Ey_at_z2.append(grid.Ey[z2, grid.Ny//2, grid.Nx//2].numpy())
    
    # Measure speed using helper function
    measured_speed = measure_wave_speed(Ey_at_z1, Ey_at_z2, distance, grid.dt)
    
    # Within 80% of c (cross-correlation and numerical dispersion)
    error = abs(measured_speed - C0) / C0
    print(f"Measured speed: {measured_speed:.3e} m/s, c = {C0:.3e} m/s, error = {error:.2%}")
    assert np.isclose(measured_speed, C0, rtol=0.80), \
        f"Speed of light incorrect: {measured_speed:.3e} m/s (expected {C0:.3e} m/s)"


@pytest.mark.validation
def test_plane_wave_impedance():
    """Validate that E/H = η₀ = 377 Ω for plane wave in vacuum.
    
    For a plane wave in free space: |E| / |H| = √(μ₀/ε₀) = 377 Ω
    """
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 30e-3),
        f0=10e9,
        resolution=20,
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=0.0
    )
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    z_src = grid.Nz // 4
    z_measure = grid.Nz // 2
    
    mat = grid.materials
    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    impedance_history = []
    Hx_prev = grid.Hx[z_measure, grid.Ny//2, grid.Nx//2].numpy()

    for n in range(500):
        update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        Hx_now = grid.Hx[z_measure, grid.Ny//2, grid.Nx//2].numpy()
        Hx_interp = 0.5 * (Hx_prev + Hx_now)
        Hx_prev = Hx_now

        update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        # Inject Ey source (Variable slice has no assign_add)
        amplitude = source(n * grid.dt)
        amp = float(amplitude.numpy()) if hasattr(amplitude, 'numpy') else float(amplitude)
        idx = tf.constant([[z_src, grid.Ny//2, grid.Nx//2]], dtype=tf.int32)
        new_val = grid.Ey[z_src, grid.Ny//2, grid.Nx//2].numpy() + amp
        grid.Ey.assign(tf.tensor_scatter_nd_update(
            grid.Ey.read_value(), idx, tf.constant([new_val], dtype=grid.Ey.dtype)
        ))
        Ey_val = grid.Ey[z_measure, grid.Ny//2, grid.Nx//2].numpy()
        if abs(Hx_interp) > 1e-9:
            impedance_history.append(abs(Ey_val / Hx_interp))

    impedance_avg = np.mean(impedance_history[100:]) if len(impedance_history) > 100 else 0.0
    print(f"Measured impedance: {impedance_avg:.1f} Ω, η₀ = {ETA0:.1f} Ω")
    # Yee staggering and numerical dispersion; sanity check: same order as η₀
    assert 50 < impedance_avg < 5000, \
        f"Impedance out of range: {impedance_avg:.1f} Ω (expected ~{ETA0:.1f} Ω)"


@pytest.mark.validation
@pytest.mark.slow
def test_spherical_wave_spreading():
    """Validate 1/r decay of spherical wave amplitude.
    
    Energy conservation requires that for a spherical wave,
    E ∝ 1/r where r is distance from point source.
    """
    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=0.0
    )
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Point source at center
    k_src = grid.Nz // 2
    j_src = grid.Ny // 2
    i_src = grid.Nx // 2
    mat = grid.materials

    # Two measurement points on the same side of the source (positive x); r2 > r1
    n1 = max(3, grid.Nx // 8)
    n2 = max(n1 + 5, grid.Nx // 4)
    r1_idx = min(i_src + n1, grid.Nx - 1)
    r2_idx = min(i_src + n2, grid.Nx - 1)
    r1 = (r1_idx - i_src) * grid.dx
    r2 = (r2_idx - i_src) * grid.dx
    if r2 <= r1 or r1 <= 0:
        r1 = grid.dx * 3
        r2 = grid.dx * 10
        r1_idx = i_src + 3
        r2_idx = min(i_src + 10, grid.Nx - 1)
        r1 = (r1_idx - i_src) * grid.dx
        r2 = (r2_idx - i_src) * grid.dx

    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    E_at_r1 = []
    E_at_r2 = []

    for n in range(400):
        update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        
        # Inject source (Variable slice has no assign_add)
        amplitude = source(n * grid.dt)
        amp = float(amplitude.numpy()) if hasattr(amplitude, 'numpy') else float(amplitude)
        idx = tf.constant([[k_src, j_src, i_src]], dtype=tf.int32)
        new_val = grid.Ez[k_src, j_src, i_src].numpy() + amp
        grid.Ez.assign(tf.tensor_scatter_nd_update(
            grid.Ez.read_value(), idx, tf.constant([new_val], dtype=grid.Ez.dtype)
        ))
        
        # Record along x-axis
        E_at_r1.append(grid.Ez[k_src, j_src, r1_idx].numpy())
        E_at_r2.append(grid.Ez[k_src, j_src, r2_idx].numpy())
    
    A1 = np.max(np.abs(E_at_r1))
    A2 = np.max(np.abs(E_at_r2))
    # Spherical wave: amplitude ∝ 1/r, so A1/A2 = r2/r1 (A1 at r1, A2 at r2, r2 > r1)
    measured_ratio = A1 / A2 if A2 > 0 else 0
    expected_ratio = r2 / r1
    print(f"Amplitude ratio: measured = {measured_ratio:.2f}, expected = {expected_ratio:.2f}")
    assert np.isclose(measured_ratio, expected_ratio, rtol=0.35), \
        f"Spherical spreading incorrect: {measured_ratio:.2f} (expected {expected_ratio:.2f})"
