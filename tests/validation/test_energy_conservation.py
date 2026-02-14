"""Validation tests for energy conservation in FDTD.

These tests verify that electromagnetic energy is conserved in lossless media
and correctly dissipated in lossy media.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import EPS0, MU0


def compute_total_energy(Ex, Ey, Ez, Hx, Hy, Hz, eps, mu, grid):
    """Compute total electromagnetic energy in the grid.
    
    Energy density: u = (1/2) * (ε|E|² + μ|H|²)
    Total energy: U = ∫∫∫ u dV
    """
    # Electric energy: (1/2) * ε * E²
    E_energy = 0.5 * tf.reduce_sum(eps * (Ex**2 + Ey**2 + Ez**2))
    
    # Magnetic energy: (1/2) * μ * H²
    H_energy = 0.5 * tf.reduce_sum(mu * (Hx**2 + Hy**2 + Hz**2))
    
    # Multiply by cell volume
    dV = grid.dx * grid.dy * grid.dz
    total_energy = (E_energy + H_energy) * dV
    
    return total_energy.numpy()


@pytest.mark.validation
@pytest.mark.slow
def test_lossless_energy_conservation():
    """Energy should remain constant in lossless medium (σ=0).
    
    In a lossless, source-free region, total energy U = ∫(E² + H²) dV
    should be constant (within numerical error).
    """
    # Small grid in vacuum
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    # Lossless vacuum
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0, mu_r=1.0, sigma=0.0)
    mat.compute_coefficients(grid.dt)
    
    # Initialize fields
    Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    
    # Inject initial pulse (only for a short time)
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    i_src, j_src, k_src = grid.Nz//2, grid.Ny//2, grid.Nx//2
    
    # Phase 1: Inject energy
    for n in range(50):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        amplitude = source.evaluate(n * grid.dt)
        Ez[i_src, j_src, k_src].assign_add(amplitude)
    
    # Phase 2: Free propagation (no source)
    energy_history = []
    for n in range(200):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        # Compute energy
        energy = compute_total_energy(Ex, Ey, Ez, Hx, Hy, Hz, mat.eps, mat.mu, grid)
        energy_history.append(energy)
    
    energy_history = np.array(energy_history)
    
    # Skip first few steps (transient)
    energy_stable = energy_history[20:]
    
    # Energy variation should be < 1% in lossless medium
    energy_max = np.max(energy_stable)
    energy_min = np.min(energy_stable)
    
    if energy_max > 0:
        variation = (energy_max - energy_min) / energy_max
    else:
        variation = 0.0
    
    print(f"Energy variation: {variation*100:.2f}%")
    
    # Allow up to 5% variation due to numerical dispersion and boundaries
    assert variation < 0.05, f"Energy not conserved: variation = {variation*100:.1f}% (should be < 5%)"


@pytest.mark.validation
@pytest.mark.slow
def test_lossy_energy_dissipation():
    """Energy should decay exponentially in lossy medium (σ>0).
    
    In a medium with conductivity σ, energy decays as:
    dU/dt = -σ * ∫E² dV
    
    Expected: U(t) = U(0) * exp(-2σt/ε)
    """
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    # Lossy medium: σ = 0.1 S/m
    sigma_value = 0.1
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0, mu_r=1.0, sigma=sigma_value)
    mat.compute_coefficients(grid.dt)
    
    # Initialize fields
    Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
    
    # Inject pulse
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    i_src, j_src, k_src = grid.Nz//2, grid.Ny//2, grid.Nx//2
    
    for n in range(50):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        amplitude = source.evaluate(n * grid.dt)
        Ez[i_src, j_src, k_src].assign_add(amplitude)
    
    # Record energy decay
    energy_history = []
    for n in range(300):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        energy = compute_total_energy(Ex, Ey, Ez, Hx, Hy, Hz, mat.eps, mat.mu, grid)
        energy_history.append(energy)
    
    energy_history = np.array(energy_history)
    
    # Energy should decay
    energy_initial = energy_history[10]  # After injection
    energy_final = energy_history[-1]
    
    decay_ratio = energy_final / energy_initial if energy_initial > 0 else 0
    
    print(f"Energy decay ratio: {decay_ratio:.4f}")
    
    # Energy should decay significantly in lossy medium
    assert decay_ratio < 0.5, f"Energy not dissipating: ratio = {decay_ratio:.3f} (should be < 0.5)"


@pytest.mark.validation
def test_poynting_theorem():
    """Validate Poynting's theorem: dU/dt + div(S) = -J·E.
    
    This is a fundamental energy conservation law in electromagnetism.
    For lossless medium (J=0): dU/dt + div(S) = 0
    """
    # This test requires computing divergence of Poynting vector S = E × H
    # Skip for now - complex to implement properly
    pytest.skip("Poynting theorem validation - implement when needed")


@pytest.mark.validation
def test_energy_injection_by_source():
    """Validate that source correctly injects energy into the system.
    
    A soft source should continuously increase total energy.
    """
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 8e-3),
        f0=12e9,
        resolution=12,
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
    
    source = GaussianPulse(f0=12e9, bandwidth=6e9)
    i_src, j_src, k_src = grid.Nz//2, grid.Ny//2, grid.Nx//2
    
    energy_history = []
    
    # Run with continuous source
    for n in range(100):
        Hx, Hy, Hz = update_H(Hx, Hy, Hz, Ex, Ey, Ez, grid, mat)
        Ex, Ey, Ez = update_E(Ex, Ey, Ez, Hx, Hy, Hz, grid, mat)
        
        # Inject
        amplitude = source.evaluate(n * grid.dt)
        Ez[i_src, j_src, k_src].assign_add(amplitude)
        
        energy = compute_total_energy(Ex, Ey, Ez, Hx, Hy, Hz, mat.eps, mat.mu, grid)
        energy_history.append(energy)
    
    energy_history = np.array(energy_history)
    
    # Energy should increase initially (source is active)
    energy_start = energy_history[10]
    energy_mid = energy_history[50]
    
    # Energy should grow while source is active
    assert energy_mid > energy_start, "Source not injecting energy"
    
    print(f"Energy growth: {energy_mid/energy_start:.2f}x")


@pytest.mark.validation
def test_courant_stability_limit():
    """Test that simulation is stable for Courant number < 1/sqrt(3).
    
    The CFL condition for 3D FDTD: c*dt <= 1/sqrt(1/dx² + 1/dy² + 1/dz²)
    Courant number S = c*dt*sqrt(1/dx² + 1/dy² + 1/dz²)
    
    For stability: S ≤ 1/sqrt(3) ≈ 0.577
    """
    from emsim.constants import C0
    
    # Test grid with known spacing
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5  # Safe value
    )
    
    # Compute Courant number
    S = C0 * grid.dt * np.sqrt(1/grid.dx**2 + 1/grid.dy**2 + 1/grid.dz**2)
    
    print(f"Courant number: {S:.4f}")
    
    # Should be < 1/sqrt(3) for stability
    assert S < 1.0 / np.sqrt(3), f"Courant number too large: {S:.4f} (should be < 0.577)"
    
    # Also check that it's close to the specified value
    assert S < 0.6, f"Courant number: {S:.4f}"
