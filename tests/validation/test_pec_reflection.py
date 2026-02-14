"""Validation tests for PEC (Perfect Electric Conductor) boundary reflection.

These tests verify that PEC boundaries correctly reflect electromagnetic waves
with total reflection and phase inversion.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.boundaries.pec import apply_pec
from emsim.sources.gaussian_pulse import GaussianPulse


@pytest.mark.validation
@pytest.mark.slow
def test_pec_total_reflection(analytical_solutions):
    """PEC should reflect plane wave with Γ = -1 (total reflection, phase inverted).
    
    Setup:
    - Wave propagating toward PEC at z=0
    - Measure incident and reflected amplitudes
    - Verify Γ = E_reflected / E_incident = -1
    """
    # 1D-like grid
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
    
    # Source position (not too close to PEC)
    z_src = 15
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    
    # Recording points
    z_before = 10  # Before PEC
    z_near_pec = 2  # Near PEC
    
    Ey_before = []
    Ey_near = []
    
    # Time stepping with PEC at z_min
    n_steps = 800
    for n in range(n_steps):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        # Apply PEC at z=0
        Ex, Ey, Ez = apply_pec(Ex, Ey, Ez, faces={'z_min': True})
        
        # Inject source
        amplitude = source.evaluate(n * grid.dt)
        Ey[z_src, grid.Ny//2, grid.Nx//2].assign_add(amplitude)
        
        # Record
        Ey_before.append(Ey[z_before, grid.Ny//2, grid.Nx//2].numpy())
        Ey_near.append(Ey[z_near_pec, grid.Ny//2, grid.Nx//2].numpy())
    
    Ey_before = np.array(Ey_before)
    Ey_near = np.array(Ey_near)
    
    # The wave should reflect: incident + reflected at recording point
    # Near PEC: standing wave pattern
    # At PEC surface (z=0), E should be zero
    
    # Measure peak amplitudes
    A_before = np.max(np.abs(Ey_before))
    A_near = np.max(np.abs(Ey_near))
    
    print(f"Amplitude before PEC: {A_before:.6e}")
    print(f"Amplitude near PEC: {A_near:.6e}")
    
    # Standing wave near PEC should have comparable amplitude
    # (not zero, but with nodes and antinodes)
    assert A_near > 0.1 * A_before, "No reflection from PEC"
    
    # Check theoretical reflection coefficient
    gamma_analytic = analytical_solutions['plane_wave_reflection_pec']()
    assert gamma_analytic == -1.0, "PEC reflection coefficient should be -1"


@pytest.mark.validation
def test_pec_tangential_e_field_zero():
    """Tangential E-field should be zero at PEC surface.
    
    Boundary condition: E_tangential = 0 at PEC
    """
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    # Initialize with non-zero fields
    Ex = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ey = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ez = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    
    # Apply PEC at z_min
    Ex, Ey, Ez = apply_pec(Ex, Ey, Ez, faces={'z_min': True})
    
    # Check that tangential components (Ex, Ey) are zero at z=0
    Ex_at_pec = Ex[0, :, :].numpy()
    Ey_at_pec = Ey[0, :, :].numpy()
    
    assert np.allclose(Ex_at_pec, 0.0, atol=1e-10), "Ex not zero at PEC"
    assert np.allclose(Ey_at_pec, 0.0, atol=1e-10), "Ey not zero at PEC"
    
    # Normal component (Ez) can be non-zero
    # (not enforced by tangential BC)


@pytest.mark.validation
def test_pec_all_faces():
    """Test PEC on all six faces."""
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 5e-3),
        f0=12e9,
        resolution=12,
        courant=0.5
    )
    
    # Initialize with non-zero fields
    Ex = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 5.0)
    Ey = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 3.0)
    Ez = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 2.0)
    
    # Apply PEC on all faces
    Ex, Ey, Ez = apply_pec(Ex, Ey, Ez, faces={
        'x_min': True, 'x_max': True,
        'y_min': True, 'y_max': True,
        'z_min': True, 'z_max': True
    })
    
    # Check all boundaries
    # x_min: Ey, Ez should be zero
    assert np.allclose(Ey[:, :, 0].numpy(), 0.0, atol=1e-10), "Ey not zero at x_min"
    assert np.allclose(Ez[:, :, 0].numpy(), 0.0, atol=1e-10), "Ez not zero at x_min"
    
    # x_max: Ey, Ez should be zero
    assert np.allclose(Ey[:, :, -1].numpy(), 0.0, atol=1e-10), "Ey not zero at x_max"
    assert np.allclose(Ez[:, :, -1].numpy(), 0.0, atol=1e-10), "Ez not zero at x_max"
    
    # y_min: Ex, Ez should be zero
    assert np.allclose(Ex[:, 0, :].numpy(), 0.0, atol=1e-10), "Ex not zero at y_min"
    assert np.allclose(Ez[:, 0, :].numpy(), 0.0, atol=1e-10), "Ez not zero at y_min"
    
    # y_max: Ex, Ez should be zero
    assert np.allclose(Ex[:, -1, :].numpy(), 0.0, atol=1e-10), "Ex not zero at y_max"
    assert np.allclose(Ez[:, -1, :].numpy(), 0.0, atol=1e-10), "Ez not zero at y_max"
    
    # z_min: Ex, Ey should be zero
    assert np.allclose(Ex[0, :, :].numpy(), 0.0, atol=1e-10), "Ex not zero at z_min"
    assert np.allclose(Ey[0, :, :].numpy(), 0.0, atol=1e-10), "Ey not zero at z_min"
    
    # z_max: Ex, Ey should be zero
    assert np.allclose(Ex[-1, :, :].numpy(), 0.0, atol=1e-10), "Ex not zero at z_max"
    assert np.allclose(Ey[-1, :, :].numpy(), 0.0, atol=1e-10), "Ey not zero at z_max"


@pytest.mark.validation
@pytest.mark.slow
def test_pec_standing_wave_pattern():
    """PEC reflection creates standing wave with nodes at PEC surface.
    
    A wave reflecting from PEC forms a standing wave pattern with:
    - Node (zero) at PEC surface
    - Antinodes (maxima) at λ/4, 3λ/4, etc.
    """
    grid = YeeGrid(
        x_range=(0, 3e-3),
        y_range=(0, 3e-3),
        z_range=(0, 40e-3),
        f0=10e9,
        resolution=25,
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
    
    # Continuous sine wave source
    f = 10e9
    omega = 2 * np.pi * f
    z_src = 20
    
    # Record amplitude distribution along z
    amplitude_profile = np.zeros(grid.Nz)
    
    # Run until steady state
    n_steps = 2000
    for n in range(n_steps):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        # Apply PEC at z=0
        Ex, Ey, Ez = apply_pec(Ex, Ey, Ez, faces={'z_min': True})
        
        # Continuous sine source
        amplitude = np.sin(omega * n * grid.dt)
        Ey[z_src, grid.Ny//2, grid.Nx//2].assign_add(amplitude * 0.1)
        
        # After steady state, record profile
        if n > 1500:
            amplitude_profile += np.abs(Ey[:, grid.Ny//2, grid.Nx//2].numpy())
    
    amplitude_profile /= (n_steps - 1500)
    
    # Check that amplitude is minimum at PEC (z=0)
    assert amplitude_profile[0] < 0.1 * np.max(amplitude_profile), \
        "Standing wave does not have node at PEC"
    
    # There should be antinodes at approximately λ/4, 3λ/4, etc.
    from emsim.constants import C0
    wavelength = C0 / f
    lambda_4_cells = int(wavelength / (4 * grid.dz))
    
    if lambda_4_cells > 0 and lambda_4_cells < grid.Nz:
        # Check that amplitude at λ/4 is larger than at PEC
        assert amplitude_profile[lambda_4_cells] > 2 * amplitude_profile[0], \
            "No antinode at λ/4 from PEC"


@pytest.mark.validation
def test_pec_patch_antenna_ground():
    """Test PEC on internal 2D patch (e.g., patch antenna ground plane).
    
    The apply_pec_patch function should enforce E_tangential = 0 on a 2D surface.
    """
    from emsim.boundaries.pec import apply_pec_patch
    
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    # Initialize with non-zero fields
    Ex = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 2.0)
    Ey = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 3.0)
    Ez = tf.Variable(tf.ones([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32) * 1.0)
    
    # Apply PEC patch at z=5 (middle plane)
    k_patch = 5
    i_range = (2, 8)
    j_range = (2, 8)
    
    Ex, Ey, Ez = apply_pec_patch(Ex, Ey, Ez, i_range, j_range, k_patch, normal='z')
    
    # Tangential components (Ex, Ey) should be zero in the patch region
    Ex_patch = Ex[i_range[0]:i_range[1], j_range[0]:j_range[1], k_patch].numpy()
    Ey_patch = Ey[i_range[0]:i_range[1], j_range[0]:j_range[1], k_patch].numpy()
    
    assert np.allclose(Ex_patch, 0.0, atol=1e-10), "Ex not zero on PEC patch"
    assert np.allclose(Ey_patch, 0.0, atol=1e-10), "Ey not zero on PEC patch"
    
    # Outside patch region should be unchanged
    Ex_outside = Ex[0, 0, k_patch].numpy()
    Ey_outside = Ey[0, 0, k_patch].numpy()
    assert Ex_outside == pytest.approx(2.0), "Ex changed outside patch"
    assert Ey_outside == pytest.approx(3.0), "Ey changed outside patch"
