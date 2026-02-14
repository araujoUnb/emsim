"""Validation tests for energy conservation in FDTD.

These tests verify that electromagnetic energy is conserved in lossless media
and correctly dissipated in lossy media.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.constants import EPS0, MU0, C0

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import run_fdtd_loop, compute_energy, assert_energy_conservation



def compute_total_energy(Ex, Ey, Ez, Hx, Hy, Hz, eps, mu, grid):
    """DEPRECATED: Use helpers.compute_energy() instead.
    
    This function is kept for backward compatibility but will be removed.
    """
    from helpers import compute_energy
    return compute_energy(grid)


@pytest.mark.validation
@pytest.mark.slow
def test_lossless_energy_conservation():
    """Energy should remain constant in lossless medium (σ=0).
    
    In a lossless, source-free region, total energy U = ∫(E² + H²) dV
    should be constant (within numerical error).
    """
    # Grid large enough for CPML (N_pml=8 requires Nx, Ny, Nz >= 16)
    grid = YeeGrid(
        x_range=(0, 40e-3),
        y_range=(0, 40e-3),
        z_range=(0, 40e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=0.0  # Lossless vacuum
    )
    
    # Source for initial energy injection
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Phase 1: Inject energy (50 steps)
    result1 = run_fdtd_loop(
        grid, n_steps=50, source=source,
        pml_faces={'x-', 'x+', 'y-', 'y+', 'z-', 'z+'}
    )
    
    # Phase 2: Free propagation (200 steps, no source)
    result2 = run_fdtd_loop(
        grid, n_steps=200, source=None,
        pml_faces={'x-', 'x+', 'y-', 'y+', 'z-', 'z+'}
    )
    
    # Check energy conservation in free propagation phase
    energy_history = result2['history']['energy']
    
    # Skip first few steps (transient)
    energy_stable = energy_history[20:]
    
    # Energy variation should be < 5% in lossless medium
    assert_energy_conservation(energy_stable, tolerance=0.05)
    
    print(f"Energy conservation test passed. {len(energy_stable)} samples checked.")


@pytest.mark.validation
@pytest.mark.slow
def test_lossy_energy_dissipation():
    """Energy should decay exponentially in lossy medium (σ>0).
    
    In a medium with conductivity σ, energy decays as:
    dU/dt = -σ * ∫E² dV
    
    Expected: U(t) = U(0) * exp(-2σt/ε)
    """
    # Lossy medium: σ = 0.1 S/m
    sigma_value = 0.1
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=sigma_value
    )
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Phase 1: Inject energy
    result1 = run_fdtd_loop(grid, n_steps=50, source=source)
    
    # Phase 2: Observe decay (no source)
    result2 = run_fdtd_loop(grid, n_steps=300, source=None)
    
    energy_history = result2['history']['energy']
    
    # Energy should decay significantly
    energy_initial = energy_history[10]  # After initial transient
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
        courant=0.5,
        eps_r=1.0, mu_r=1.0, sigma=0.0
    )
    
    source = GaussianPulse(f0=12e9, bandwidth=6e9)
    
    # Run with continuous source
    result = run_fdtd_loop(grid, n_steps=100, source=source)
    
    energy_history = result['history']['energy']
    
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
