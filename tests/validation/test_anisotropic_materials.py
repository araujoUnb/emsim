"""Validation tests for anisotropic (diagonal tensor) materials in FDTD.

Checks solver stability with uniaxial anisotropic regions. Documentation in English.
"""

import numpy as np
import pytest

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.materials.base import AnisotropicMaterial


def test_anisotropic_solver_stable():
    """Run FDTD with an anisotropic (uniaxial) region; fields must remain finite."""
    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=12,
    )
    ni, nj, nk = grid.Nx, grid.Ny, grid.Nz
    mat = AnisotropicMaterial(
        name="Uniaxial test",
        eps_r=1.0,
        eps_r_xx=3.0,
        eps_r_yy=3.0,
        eps_r_zz=2.0,
    )
    grid.materials.add_anisotropic_region(
        (2, ni - 2), (2, nj - 2), (2, nk - 2),
        material=mat,
    )
    grid.materials.compute_coefficients(grid.dt)
    source = GaussianPulse(f0=10e9, bandwidth=2e9)
    solver = FDTDSolver(grid=grid, source=source, pml_faces=set(), n_steps=150)
    assert solver.has_dispersive is True  # branch includes anisotropic
    solver.run(verbose=False)
    E = np.concatenate([
        grid.Ex.numpy().ravel(),
        grid.Ey.numpy().ravel(),
        grid.Ez.numpy().ravel(),
    ])
    assert np.all(np.isfinite(E)), "E-field must be finite after anisotropic run"


def test_anisotropic_material_creation():
    """AnisotropicMaterial has expected diagonal tensor attributes."""
    mat = AnisotropicMaterial(
        name="LC",
        eps_r=1.0,
        eps_r_xx=3.0,
        eps_r_yy=3.0,
        eps_r_zz=2.0,
    )
    assert mat.anisotropic is True
    assert mat.eps_r_xx == 3.0
    assert mat.eps_r_zz == 2.0
