"""Validation tests for CPML (Convolutional Perfectly Matched Layer) absorption.

Tests run through FDTDSolver with pml_faces to ensure CPML is applied and
simulations remain stable. Documentation in English.
"""

import pytest
import numpy as np

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


@pytest.mark.validation
def test_cpml_solver_stable():
    """Run FDTD with CPML on z faces; fields must remain finite (no explosion)."""
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 40e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
    )
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=8,
        n_steps=500,
    )
    result = solver.run(verbose=False)
    assert result is not None
    Ey = grid.Ey.numpy()
    assert np.all(np.isfinite(Ey)), "Fields must be finite with CPML"
    assert np.max(np.abs(Ey)) < 1e8, "Field magnitude should be bounded"


@pytest.mark.validation
def test_cpml_creation_via_solver():
    """CPML is created and used when pml_faces is set."""
    grid = YeeGrid(
        x_range=(0, 8e-3),
        y_range=(0, 8e-3),
        z_range=(0, 30e-3),
        f0=10e9,
        resolution=12,
    )
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=6,
        n_steps=200,
    )
    assert solver.cpml is not None
    result = solver.run(verbose=False)
    assert "n_steps_run" in result
