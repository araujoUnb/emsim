"""Integration tests for solver with boundary conditions.

These tests verify that the FDTD solver correctly integrates with PEC and CPML
boundaries using the current FDTDSolver API (pml_faces, pec_faces, pec_regions,
ports for source injection). Documentation in English.
"""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


@pytest.mark.integration
def test_solver_with_cpml():
    """Test solver with CPML boundaries; run completes and fields stay finite."""
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
    )
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    # Inject at center via lumped port so fields are non-zero
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=8,
        n_steps=300,
    )
    result = solver.run(verbose=False)
    assert result is not None
    assert "n_steps_run" in result
    Ex = grid.Ex.numpy()
    assert np.all(np.isfinite(Ex)), "E-field must be finite"
    assert not np.any(np.isnan(Ex)) and not np.any(np.isinf(Ex))


@pytest.mark.integration
def test_solver_with_pec_regions():
    """Test solver with internal PEC regions (patch)."""
    grid = YeeGrid(
        x_range=(0, 15e-3),
        y_range=(0, 15e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=12,
        courant=0.5,
    )
    pec_regions = [
        {"i_range": (5, 10), "j_range": (5, 10), "k": 2, "normal": "z"},
    ]
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pec_regions=pec_regions,
        pml_faces=set(),
        n_steps=200,
    )
    result = solver.run(verbose=False)
    assert result is not None
    # PEC patch: tangential E (Ex, Ey) at k=2 in patch should be ~0
    Ex = grid.Ex.numpy()
    Ey = grid.Ey.numpy()
    Ex_pec = Ex[2, 5:10, 5:10]
    Ey_pec = Ey[2, 5:10, 5:10]
    assert np.max(np.abs(Ex_pec)) < 1e-4, "PEC should zero Ex on patch"
    assert np.max(np.abs(Ey_pec)) < 1e-4, "PEC should zero Ey on patch"


@pytest.mark.integration
def test_solver_with_pec_and_cpml():
    """Test solver with both PEC regions and CPML on z only (larger grid for PML)."""
    grid = YeeGrid(
        x_range=(0, 15e-3),
        y_range=(0, 15e-3),
        z_range=(0, 25e-3),
        f0=12e9,
        resolution=10,
        courant=0.5,
    )
    pec_regions = [
        {"i_range": (3, 6), "j_range": (3, 9), "k": 4, "normal": "z"},
    ]
    source = GaussianPulse(f0=12e9, bandwidth=6e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=6,
        pec_regions=pec_regions,
        n_steps=250,
    )
    result = solver.run(verbose=False)
    assert result is not None
    assert np.all(np.isfinite(grid.Ey.numpy())), "Ey must be finite"


@pytest.mark.integration
def test_solver_boundary_stability():
    """Test that solver remains stable with CPML on z-boundaries."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 16e-3),
        f0=15e9,
        resolution=12,
        courant=0.5,
    )
    source = GaussianPulse(f0=15e9, bandwidth=7e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=8,
        n_steps=400,
    )
    result = solver.run(verbose=False)
    assert result is not None
    Ez = grid.Ez.numpy()
    max_Ez = np.max(np.abs(Ez))
    assert np.isfinite(max_Ez), "Field became infinite (unstable)"
    assert max_Ez < 1e10, "Field too large (possible instability)"


@pytest.mark.integration
def test_solver_pec_external_faces():
    """Test solver with PEC on external faces (cavity)."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 6e-3),
        z_range=(0, 5e-3),
        f0=20e9,
        resolution=15,
        courant=0.5,
    )
    source = GaussianPulse(f0=20e9, bandwidth=10e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pec_faces={"x-", "x+", "y-", "y+", "z-", "z+"},
        n_steps=200,
    )
    result = solver.run(verbose=False)
    assert result is not None
