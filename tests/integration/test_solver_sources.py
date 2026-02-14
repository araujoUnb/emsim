"""Integration tests for solver with different source types.

These tests verify that the FDTD solver correctly integrates with various
source configurations (soft sources, modal ports, lumped ports).
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


@pytest.mark.integration
def test_solver_with_soft_source(small_grid):
    """Test that solver runs with a simple soft source injection via LumpedPort."""
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    port = LumpedPort(
        name="src",
        i=small_grid.Nx//2,
        j=small_grid.Ny//2,
        k=small_grid.Nz//2,
        direction='z',
        resistance=50.0
    )
    solver = FDTDSolver(
        grid=small_grid,
        source=source,
        ports=[port],
        n_steps=100
    )
    result = solver.run()
    assert result is not None
    assert 'n_steps_run' in result
    assert result['n_steps_run'] == 100
    # Fields live on grid after run
    assert tf.reduce_sum(tf.abs(small_grid.Ez)).numpy() >= 0


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
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    port = LumpedPort(
        name="port1",
        i=grid.Nx//2,
        j=grid.Ny//2,
        k=grid.Nz//2,
        direction='z',
        resistance=50.0
    )
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        n_steps=200
    )
    result = solver.run()
    assert result is not None
    assert 'n_steps_run' in result
    assert 'Z_in' in result
    assert 'S11' in result


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
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    port1 = LumpedPort(
        name="input",
        i=grid.Nx//2,
        j=grid.Ny//2,
        k=10,
        direction='z',
        resistance=50.0
    )
    port2 = LumpedPort(
        name="output",
        i=grid.Nx//2,
        j=grid.Ny//2,
        k=grid.Nz - 10,
        direction='z',
        resistance=50.0
    )
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port1, port2],
        n_steps=300
    )
    result = solver.run()
    assert result is not None
    assert result['n_steps_run'] == 300
    # Solver returns last port's S11/Z_in; both ports were run
    assert 'S11' in result


@pytest.mark.integration
def test_solver_no_source_no_crash():
    """Test that solver doesn't crash with no ports (waveform computed but no injection)."""
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=10,
        courant=0.5
    )
    # Source required by solver API; with ports=None nothing is injected
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=None,
        n_steps=50
    )
    result = solver.run()
    assert result is not None
    assert result['n_steps_run'] == 50


@pytest.mark.integration
def test_solver_source_timing():
    """Test that source injection via port happens and fields are non-zero."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 16e-3),
        f0=10e9,
        resolution=12,
        courant=0.5
    )
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    port = LumpedPort(
        name="src",
        i=grid.Nx//2,
        j=grid.Ny//2,
        k=grid.Nz//2,
        direction='z',
        resistance=50.0
    )
    source_duration = int(5 / (5e9) / grid.dt)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        n_steps=source_duration + 100
    )
    result = solver.run()
    # Fields on grid after run
    Ex_final = grid.Ex
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
    # Add substrate region with eps_r = 2.5 (i,j,k = Nx, Ny, Nz indices)
    grid.materials.set_region(
        i_range=(0, grid.Nx),
        j_range=(0, grid.Ny),
        k_range=(0, min(3, grid.Nz)),
        eps_r=2.5
    )
    grid.materials.compute_coefficients(grid.dt)
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    port = LumpedPort(
        name="src",
        i=grid.Nx//2,
        j=grid.Ny//2,
        k=min(5, grid.Nz - 1),
        direction='z',
        resistance=50.0
    )
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        n_steps=200
    )
    result = solver.run()
    assert result is not None
    Ez = grid.Ez
    # Substrate in k=0..2; air in upper half
    k_sub = min(1, grid.Nz - 1)
    k_air = max(grid.Nz - 2, 0)
    Ez_substrate = Ez[k_sub, grid.Ny//2, grid.Nx//2].numpy()
    Ez_air = Ez[k_air, grid.Ny//2, grid.Nx//2].numpy()
    assert not np.isnan(Ez_substrate)
    assert not np.isnan(Ez_air)


@pytest.mark.integration
def test_solver_nonuniform_grid_runs():
    """Solver with non-uniform (stretched) grid runs without error; fields remain finite."""
    # dz finer in center, coarser at ends (e.g. for local refinement)
    Nz = 30
    dz_outer = 0.001
    dz_inner = 0.0005
    dz_arr = [dz_outer] * 10 + [dz_inner] * 10 + [dz_outer] * 10
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 30e-3),
        dx=0.0005,
        dy=0.0005,
        dz=dz_arr,
    )
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    port = LumpedPort(
        name="src",
        i=grid.Nx // 2,
        j=grid.Ny // 2,
        k=grid.Nz // 2,
        direction="z",
        resistance=50.0,
    )
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        n_steps=50,
    )
    result = solver.run(verbose=False)
    assert result is not None
    assert result["n_steps_run"] == 50
    # Fields must be finite
    for comp in (grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz):
        assert tf.reduce_all(tf.math.is_finite(comp)).numpy(), "Non-finite field component"
