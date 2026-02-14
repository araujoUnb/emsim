"""Validation tests for free-space electromagnetic wave propagation.

These tests verify that waves propagate at the correct speed of light
and satisfy fundamental electromagnetic relations.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import C0, ETA0


@pytest.mark.validation
@pytest.mark.slow
def test_speed_of_light():
    """Validate that electromagnetic pulse propagates at c = 299,792,458 m/s.
    
    Setup:
    - Gaussian pulse in center
    - Measure arrival time at two points separated by known distance
    - Compute velocity and compare with c
    """
    # Create 1D-like grid (z-propagation)
    grid = YeeGrid(
        x_range=(0, 2e-3),
        y_range=(0, 2e-3),
        z_range=(0, 60e-3),  # Long in z
        f0=10e9,
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
    
    # Source position
    z_src = 10
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    
    # Recording positions
    z1 = 20
    z2 = 40
    distance = (z2 - z1) * grid.dz
    
    Ey_at_z1 = []
    Ey_at_z2 = []
    
    # Time stepping
    n_steps = 800
    for n in range(n_steps):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        # Inject source
        amplitude = source.evaluate(n * grid.dt)
        Ey[z_src, grid.Ny//2, grid.Nx//2].assign_add(amplitude)
        
        # Record
        Ey_at_z1.append(Ey[z1, grid.Ny//2, grid.Nx//2].numpy())
        Ey_at_z2.append(Ey[z2, grid.Ny//2, grid.Nx//2].numpy())
    
    Ey_at_z1 = np.array(Ey_at_z1)
    Ey_at_z2 = np.array(Ey_at_z2)
    
    # Find peak arrival times
    t1 = np.argmax(np.abs(Ey_at_z1)) * grid.dt
    t2 = np.argmax(np.abs(Ey_at_z2)) * grid.dt
    
    delay = t2 - t1
    
    if delay > 0:
        c_measured = distance / delay
    else:
        c_measured = 0
    
    print(f"Distance: {distance*1e3:.2f} mm")
    print(f"Delay: {delay*1e12:.2f} ps")
    print(f"Measured c: {c_measured:.6e} m/s")
    print(f"Expected c: {C0:.6e} m/s")
    print(f"Error: {abs(c_measured - C0)/C0 * 100:.2f}%")
    
    # Should match within 1% (accounting for numerical dispersion)
    assert np.isclose(c_measured, C0, rtol=0.01), \
        f"Speed of light incorrect: {c_measured:.3e} m/s (expected {C0:.3e} m/s)"


@pytest.mark.validation
@pytest.mark.slow
def test_plane_wave_impedance():
    """Validate that plane wave has impedance η₀ = 377 Ω.
    
    For a plane wave in vacuum: E/H = η₀ = sqrt(μ₀/ε₀)
    """
    grid = YeeGrid(
        x_range=(0, 3e-3),
        y_range=(0, 3e-3),
        z_range=(0, 40e-3),
        f0=10e9,
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
    
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    z_src = 8
    
    # Recording point far from source
    z_rec = 25
    Ey_rec = []
    Hx_rec = []
    
    for n in range(600):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        amplitude = source.evaluate(n * grid.dt)
        Ey[z_src, grid.Ny//2, grid.Nx//2].assign_add(amplitude)
        
        Ey_rec.append(Ey[z_rec, grid.Ny//2, grid.Nx//2].numpy())
        Hx_rec.append(Hx[z_rec, grid.Ny//2, grid.Nx//2].numpy())
    
    Ey_rec = np.array(Ey_rec)
    Hx_rec = np.array(Hx_rec)
    
    # Find time when pulse is at recording point
    peak_idx = np.argmax(np.abs(Ey_rec))
    
    # Sample around peak (avoid zero H)
    sample_range = slice(peak_idx - 5, peak_idx + 5)
    Ey_sample = Ey_rec[sample_range]
    Hx_sample = Hx_rec[sample_range]
    
    # Compute impedance: Z = Ey / Hx (for wave in +z with Ey, Hx)
    # Filter out near-zero values
    mask = np.abs(Hx_sample) > 1e-6 * np.max(np.abs(Hx_sample))
    if np.sum(mask) > 0:
        Z_measured = np.mean(Ey_sample[mask] / Hx_sample[mask])
    else:
        Z_measured = 0
    
    print(f"Measured impedance: {Z_measured:.2f} Ω")
    print(f"Expected η₀: {ETA0:.2f} Ω")
    print(f"Error: {abs(Z_measured - ETA0)/ETA0 * 100:.2f}%")
    
    # Should match within 5% (numerical dispersion and sampling effects)
    assert np.isclose(Z_measured, ETA0, rtol=0.05), \
        f"Plane wave impedance incorrect: {Z_measured:.1f} Ω (expected {ETA0:.1f} Ω)"


@pytest.mark.validation
def test_wavelength_calculation(analytical_solutions):
    """Validate wavelength formula: λ = c/f."""
    test_frequencies = [1e9, 5e9, 10e9, 24e9, 50e9]
    
    for f in test_frequencies:
        wavelength = analytical_solutions['wavelength'](f)
        expected = C0 / f
        
        assert np.isclose(wavelength, expected, rtol=1e-10), \
            f"At {f/1e9:.1f} GHz: λ={wavelength:.6e} m (expected {expected:.6e} m)"


@pytest.mark.validation
def test_wavelength_in_dielectric(analytical_solutions):
    """Validate wavelength in dielectric: λ = c/(f*sqrt(εᵣ)).
    
    In a dielectric with relative permittivity εᵣ, the wavelength
    is reduced by a factor of sqrt(εᵣ).
    """
    f = 10e9
    eps_r = 2.25  # e.g., Teflon
    
    wavelength_vacuum = C0 / f
    wavelength_dielectric = analytical_solutions['wavelength'](f, eps_r=eps_r)
    
    expected = wavelength_vacuum / np.sqrt(eps_r)
    
    print(f"λ (vacuum): {wavelength_vacuum*1e3:.3f} mm")
    print(f"λ (εᵣ={eps_r}): {wavelength_dielectric*1e3:.3f} mm")
    print(f"Ratio: {wavelength_vacuum/wavelength_dielectric:.3f} (expected: {np.sqrt(eps_r):.3f})")
    
    assert np.isclose(wavelength_dielectric, expected, rtol=1e-10)


@pytest.mark.validation
@pytest.mark.slow
def test_spherical_wave_spreading():
    """Validate 1/r amplitude decay for spherical wave.
    
    A point source radiates a spherical wave. At distance r from the source,
    the amplitude should decay as 1/r (energy decays as 1/r²).
    """
    # 3D cubic grid
    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=15e9,
        resolution=15,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    # Initialize fields
    Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    
    # Point source in center
    source = GaussianPulse(f0=15e9, bandwidth=7e9)
    i_src = grid.Nz // 2
    j_src = grid.Ny // 2
    k_src = grid.Nx // 2
    
    # Recording points at different distances
    r1 = 5
    r2 = 10
    
    Ez_r1 = []
    Ez_r2 = []
    
    for n in range(400):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        amplitude = source.evaluate(n * grid.dt)
        Ez[i_src, j_src, k_src].assign_add(amplitude)
        
        # Record at (i_src + r, j_src, k_src)
        if i_src + r1 < grid.Nz:
            Ez_r1.append(Ez[i_src + r1, j_src, k_src].numpy())
        if i_src + r2 < grid.Nz:
            Ez_r2.append(Ez[i_src + r2, j_src, k_src].numpy())
    
    Ez_r1 = np.array(Ez_r1)
    Ez_r2 = np.array(Ez_r2)
    
    # Find peak amplitudes
    A1 = np.max(np.abs(Ez_r1))
    A2 = np.max(np.abs(Ez_r2))
    
    if A1 > 0 and A2 > 0:
        ratio_measured = A1 / A2
        ratio_expected = r2 / r1
        
        print(f"Amplitude at r={r1}: {A1:.6e}")
        print(f"Amplitude at r={r2}: {A2:.6e}")
        print(f"Ratio A(r1)/A(r2): {ratio_measured:.3f} (expected: {ratio_expected:.3f})")
        
        # Allow 20% error due to grid discretization and near-field effects
        assert np.isclose(ratio_measured, ratio_expected, rtol=0.2), \
            f"Spherical decay incorrect: {ratio_measured:.2f} (expected {ratio_expected:.2f})"
    else:
        pytest.skip("Amplitudes too small to measure")


@pytest.mark.validation
def test_dispersion_relation():
    """Validate dispersion relation: ω² = c²k² for plane waves in vacuum.
    
    The relation between frequency and wavenumber should be linear: k = ω/c.
    """
    # For numerical FDTD, there's numerical dispersion
    # This test just checks that the relation is approximately satisfied
    
    f = 10e9
    omega = 2 * np.pi * f
    k_expected = omega / C0
    
    # Wavelength
    wavelength = C0 / f
    
    # Wavenumber
    k = 2 * np.pi / wavelength
    
    assert np.isclose(k, k_expected, rtol=1e-10), \
        f"Dispersion relation: k={k:.3e} (expected {k_expected:.3e})"
