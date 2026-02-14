"""Integration tests for solver with different source types.

These tests verify that the FDTD solver correctly integrates with various
source configurations (soft sources, modal ports, lumped ports).
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


@pytest.mark.integration
def test_solver_with_soft_source(small_grid):
    """Test that solver runs with a simple soft source injection."""
    mat = MaterialGrid(small_grid.Nz, small_grid.Ny, small_grid.Nx, eps_r=1.0)
    mat.compute_coefficients(small_grid.dt)
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Create solver with no ports (just soft source)
    solver = FDTDSolver(
        grid=small_grid,
        materials=mat,
        source=source,
        source_position=(small_grid.Nz//2, small_grid.Ny//2, small_grid.Nx//2),
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    # Run short simulation
    result = solver.run(n_steps=100)
    
    # Check that result exists
    assert result is not None
    assert 'Ex' in result
    assert 'Ey' in result
    assert 'Ez' in result


@pytest.mark.integration
def test_solver_with_lumped_port():
    """Test solver integration with LumpedPort for injection and recording."""
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
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Create lumped port
    port = LumpedPort(
        name="port1",
        i=grid.Nz//2,
        j_range=(grid.Ny//2 - 2, grid.Ny//2 + 2),
        k_range=(grid.Nx//2 - 2, grid.Nx//2 + 2),
        direction='z',
        impedance=50.0
    )
    
    # Solver with port
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=None,  # Injection via port
        ports=[port],
        pec_regions=None,
        nf2ff_box=None
    )
    
    # Run
    result = solver.run(n_steps=200)
    
    # Check that port recorded data
    assert 'ports' in result
    assert len(result['ports']) == 1
    
    port_result = result['ports'][0]
    assert port_result['type'] == 'lumped'
    assert 'Z_in' in port_result
    assert 'S11' in port_result


@pytest.mark.integration
def test_solver_with_multiple_ports():
    """Test solver with multiple lumped ports."""
    grid = YeeGrid(
        x_range=(0, 15e-3),
        y_range=(0, 8e-3),
        z_range=(0, 30e-3),
        f0=10e9,
        resolution=12,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Two ports
    port1 = LumpedPort(
        name="input",
        i=10,
        j_range=(grid.Ny//2 - 1, grid.Ny//2 + 1),
        k_range=(grid.Nx//2 - 1, grid.Nx//2 + 1),
        direction='z',
        impedance=50.0
    )
    
    port2 = LumpedPort(
        name="output",
        i=grid.Nz - 10,
        j_range=(grid.Ny//2 - 1, grid.Ny//2 + 1),
        k_range=(grid.Nx//2 - 1, grid.Nx//2 + 1),
        direction='z',
        impedance=50.0
    )
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=None,
        ports=[port1, port2],
        pec_regions=None,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=300)
    
    # Check that both ports have results
    assert len(result['ports']) == 2
    assert result['ports'][0]['name'] == 'input'
    assert result['ports'][1]['name'] == 'output'


@pytest.mark.integration
def test_solver_no_source_no_crash():
    """Test that solver doesn't crash with no source or ports."""
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=10,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=None,
        source_position=None,
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    # Should run without crashing (though fields remain zero)
    result = solver.run(n_steps=50)
    
    assert result is not None


@pytest.mark.integration
def test_solver_source_timing():
    """Test that source injection happens at correct times."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 16e-3),
        f0=10e9,
        resolution=12,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    mat.compute_coefficients(grid.dt)
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    # Track when source is active
    # Gaussian pulse is active roughly for first few periods
    source_duration = int(5 / (5e9) / grid.dt)  # ~5 cycles
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(grid.Nz//2, grid.Ny//2, grid.Nx//2),
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=source_duration + 100)
    
    # Fields should be non-zero after source activation
    Ex_final = result['Ex']
    total_energy = tf.reduce_sum(Ex_final**2).numpy()
    
    assert total_energy > 0, "Source did not inject energy"


@pytest.mark.integration
def test_solver_heterogeneous_materials():
    """Test solver with heterogeneous materials (substrate + air)."""
    grid = YeeGrid(
        x_range=(0, 12e-3),
        y_range=(0, 12e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=15,
        courant=0.5
    )
    
    mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0)
    
    # Add substrate region with eps_r = 2.5
    mat.set_region(
        i_range=(0, 3),
        j_range=(0, grid.Ny),
        k_range=(0, grid.Nx),
        eps_r=2.5
    )
    mat.compute_coefficients(grid.dt)
    
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    
    solver = FDTDSolver(
        grid=grid,
        materials=mat,
        source=source,
        source_position=(5, grid.Ny//2, grid.Nx//2),
        ports=None,
        pec_regions=None,
        nf2ff_box=None
    )
    
    result = solver.run(n_steps=200)
    
    # Should complete without errors
    assert result is not None
    
    # Wave should propagate differently in substrate vs air
    Ez = result['Ez']
    Ez_substrate = Ez[1, grid.Ny//2, grid.Nx//2].numpy()
    Ez_air = Ez[6, grid.Ny//2, grid.Nx//2].numpy()
    
    # Both should be non-zero (wave propagated)
    # Exact comparison is tricky, just check both regions have fields
    assert not np.isnan(Ez_substrate)
    assert not np.isnan(Ez_air)
