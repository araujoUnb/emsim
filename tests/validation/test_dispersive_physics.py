"""Validation tests for dispersive (Drude) materials in FDTD.

Checks solver stability with Drude regions and that auxiliary fields P, J
evolve as expected. Documentation in English.
"""

import numpy as np
import pytest

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.materials import get_material_manager


def test_drude_solver_stable():
    """Run FDTD with a Drude (copper) region; fields must remain finite (no NaN/Inf)."""
    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=12,
    )
    # Region within grid bounds (e.g. Nx ~ 8 for 20mm @ res 12)
    ni, nj, nk = grid.Nx, grid.Ny, grid.Nz
    mgr = get_material_manager()
    mgr.apply_to_grid(
        grid,
        region={"i": (2, ni - 2), "j": (2, nj - 2), "k": (2, nk - 2)},
        material_name="copper_drude",
    )
    grid.materials.compute_coefficients(grid.dt)
    source = GaussianPulse(f0=10e9, bandwidth=3e9)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        pml_faces=set(),
        n_steps=200,
    )
    assert solver.has_dispersive is True
    solver.run(verbose=False)
    E = np.concatenate([
        grid.Ex.numpy().ravel(),
        grid.Ey.numpy().ravel(),
        grid.Ez.numpy().ravel(),
    ])
    assert np.all(np.isfinite(E)), "E-field must be finite after Drude run"
    assert not np.any(np.isnan(E)), "E-field must not contain NaN"


def test_drude_auxiliary_fields_evolve():
    """With Drude region, P and J should be non-zero inside the region after some steps."""
    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=12,
    )
    ni, nj, nk = grid.Nx, grid.Ny, grid.Nz
    mgr = get_material_manager()
    mgr.apply_to_grid(
        grid,
        region={"i": (2, ni - 2), "j": (2, nj - 2), "k": (2, nk - 2)},
        material_name="copper_drude",
    )
    grid.materials.compute_coefficients(grid.dt)
    source = GaussianPulse(f0=10e9, bandwidth=2e9)
    solver = FDTDSolver(grid=grid, source=source, pml_faces=set(), n_steps=100)
    solver.run(verbose=False)
    # Auxiliary arrays must exist and remain finite (no NaN/Inf)
    Pz = grid.materials.Pz.numpy()
    Jz = grid.materials.Jz.numpy()
    assert np.all(np.isfinite(Pz)) and np.all(np.isfinite(Jz)), (
        "P and J must be finite after Drude run"
    )


def test_drude_material_creation():
    """DispersiveMaterial with model drude has expected attributes."""
    from emsim.materials.base import DispersiveMaterial

    mat = DispersiveMaterial(
        name="Silver",
        eps_r=1.0,
        model="drude",
        eps_inf=3.7,
        omega_p=1.38e16,
        gamma=2.73e13,
    )
    assert mat.dispersive is True
    assert mat.model == "drude"
    assert mat.omega_p == 1.38e16
    assert mat.gamma == 2.73e13
