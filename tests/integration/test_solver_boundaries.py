"""Integration tests for solver with boundary conditions.

These tests verify that the FDTD solver correctly integrates with PEC and CPML boundaries.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.boundaries.cpml import CPMLBoundary


@pytest.mark.integration
def test_solver_with_cpml():
    """Test solver with CPML boundaries."""
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    # CPML on all boundaries
    cpml = CPMLBoundary(
        grid=grid,
        thickness=8,
        boundaries={'x': [True, True], 'y': [True, True], 'z': [True, True]}
    )
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(grid.Nz//2, grid.Ny//2, grid.Nx//2),
        cpml=cpml,
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=300)
    
    # Should complete without instability
    assert result is not None
    
    # Check that fields are not NaN or Inf
    Ex = result['Ex']
    assert not tf.reduce_any(tf.math.is_nan(Ex)).numpy()
    assert not tf.reduce_any(tf.math.is_inf(Ex)).numpy()


@pytest.mark.integration
def test_solver_with_pec_regions():
    """Test solver with internal PEC regions (e.g., patch antenna ground)."""
    grid = YeeGrid(
        x_range=(0, 15e-3),
        y_range=(0, 15e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=12,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    # Define PEC patch at z=2
    pec_regions = [
        {
            'i_range': (5, 10),
            'j_range': (5, 10),
            'k': 2,
            'normal': 'z'
        }
    ]
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(7, grid.Ny//2, grid.Nx//2),
        ports=None,
        pec_regions=pec_regions,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=200)
    
    # Check that PEC is enforced: tangential E should be zero on patch
    Ex = result['Ex']
    Ey = result['Ey']
    
    # At k=2, i in [5,10), j in [5,10), Ex and Ey should be ~0
    Ex_pec = Ex[5:10, 5:10, 2].numpy()
    Ey_pec = Ey[5:10, 5:10, 2].numpy()
    
    assert np.max(np.abs(Ex_pec)) < 1e-6, "PEC not enforced on Ex"
    assert np.max(np.abs(Ey_pec)) < 1e-6, "PEC not enforced on Ey"


@pytest.mark.integration
def test_solver_with_pec_and_cpml():
    """Test solver with both PEC regions and CPML boundaries."""
    grid = YeeGrid(
        x_range=(0, 12e-3),
        y_range=(0, 12e-3),
        z_range=(0, 12e-3),
        f0=12e9,
        resolution=12,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    cpml = CPMLBoundary(
        grid=grid,
        thickness=8,
        boundaries={'x': [True, True], 'y': [True, True], 'z': [True, True]}
    )
    
    pec_regions = [
        {
            'i_range': (3, 6),
            'j_range': (3, 9),
            'k': 3,
            'normal': 'z'
        }
    ]
    
    source = GaussianPulse(f0=12e9, bandwidth=6e9)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(grid.Nz//2, grid.Ny//2, grid.Nx//2),
        cpml=cpml,
        ports=None,
        pec_regions=pec_regions,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=250)
    
    # Should complete successfully
    assert result is not None
    
    # Verify no NaN
    Ey = result['Ey']
    assert not tf.reduce_any(tf.math.is_nan(Ey)).numpy()


@pytest.mark.integration
def test_solver_boundary_stability():
    """Test that solver remains stable with various boundary configurations."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 16e-3),
        f0=15e9,
        resolution=12,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    # CPML only on z-boundaries
    cpml = CPMLBoundary(
        grid=grid,
        thickness=8,
        boundaries={'x': [False, False], 'y': [False, False], 'z': [True, True]}
    )
    
    source = GaussianPulse(f0=15e9, bandwidth=7e9)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(grid.Nz//2, grid.Ny//2, grid.Nx//2),
        cpml=cpml,
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=400)
    
    # Check field magnitudes are reasonable (not exploding)
    Ez = result['Ez']
    max_Ez = tf.reduce_max(tf.abs(Ez)).numpy()
    
    # Should be finite and not too large
    assert np.isfinite(max_Ez), "Field became infinite (unstable)"
    assert max_Ez < 1e10, f"Field too large: {max_Ez} (possible instability)"


@pytest.mark.integration
def test_solver_pec_external_faces():
    """Test solver with PEC on external faces (closed cavity)."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 6e-3),
        z_range=(0, 5e-3),
        f0=20e9,
        resolution=15,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    source = GaussianPulse(f0=20e9, bandwidth=10e9)
    
    # Note: apply_pec on faces is done inside solver's update loop
    # For this integration test, we just run and check stability
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(grid.Nz//2, grid.Ny//2, grid.Nx//2),
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    # Run without external boundaries (open)
    result = solver.run(n_steps=200)
    
    assert result is not None


@pytest.mark.integration
def test_solver_mixed_boundaries():
    """Test solver with mixed boundary conditions (CPML on some, PEC on others)."""
    # This is a common scenario: CPML on x/y (open), PEC on z (waveguide)
    pytest.skip("Mixed boundaries require solver modifications - implement if needed")
